// GOL-on-ANE: Tiled Game of Life on Apple Neural Engine (Swift)
//
// Loads a programmed GOL computer (~1.4M cells in ~210K×215K grid),
// tiles it into 1024×1024 chunks, and runs each tile through CoreML/ANE.
//
// The ANE executes Conway's B3/S23 rules via Conv2d — no cheating.
// Chain: ANE (Conv2d) → GOL (B3/S23) → Computer → Matmul
//
// Build:
//   swiftc -O -framework CoreML -framework Accelerate -o gol_ane main.swift
//
// Usage:
//   ./gol_ane [--gens N] [--cpu-only] [--cells gol_computer.bin]

import CoreML
import Foundation
import Accelerate
import Dispatch

// MARK: - Configuration

let TILE_SIZE = 1024
let PADDED    = TILE_SIZE + 2  // 1026

// Float16 constants as UInt16 bit patterns
let F16_ONE:  UInt16 = 0x3C00  // 1.0 in Float16
let F16_ZERO: UInt16 = 0x0000  // 0.0 in Float16
// For alive check: anything >= 0x3800 (0.5 in Float16)
let F16_HALF: UInt16 = 0x3800

@inline(__always)
func isAlive(_ v: UInt16) -> Bool { v >= F16_HALF }

// MARK: - Tile Key

struct TileKey: Hashable {
    let row: Int
    let col: Int
}

// MARK: - Tiled GOL Grid (Float16 / UInt16 storage)

final class TiledGOL {
    let tileSize: Int
    var tiles: [TileKey: UnsafeMutableBufferPointer<UInt16>]
    var rowOffset: Int = 0
    var colOffset: Int = 0
    var generation: UInt64 = 0

    init(tileSize: Int = TILE_SIZE) {
        self.tileSize = tileSize
        self.tiles = [:]
    }

    deinit {
        for (_, buf) in tiles {
            buf.baseAddress!.deallocate()
        }
    }

    private func allocTile() -> UnsafeMutableBufferPointer<UInt16> {
        let count = tileSize * tileSize
        let ptr = UnsafeMutablePointer<UInt16>.allocate(capacity: count)
        ptr.initialize(repeating: F16_ZERO, count: count)
        return UnsafeMutableBufferPointer(start: ptr, count: count)
    }

    func setCell(_ row: Int, _ col: Int) {
        let lr = row - rowOffset
        let lc = col - colOffset
        let tr = lr >= 0 ? lr / tileSize : (lr - tileSize + 1) / tileSize
        let tc = lc >= 0 ? lc / tileSize : (lc - tileSize + 1) / tileSize
        let localR = lr - tr * tileSize
        let localC = lc - tc * tileSize
        let key = TileKey(row: tr, col: tc)
        if tiles[key] == nil {
            tiles[key] = allocTile()
        }
        tiles[key]![localR * tileSize + localC] = F16_ONE
    }

    func population() -> Int {
        var total = 0
        for (_, buf) in tiles {
            let base = buf.baseAddress!
            for i in 0..<buf.count {
                if isAlive(base[i]) { total += 1 }
            }
        }
        return total
    }

    func activeTiles() -> Set<TileKey> {
        var active = Set<TileKey>()
        for (key, buf) in tiles {
            let base = buf.baseAddress!
            for i in 0..<buf.count {
                if isAlive(base[i]) { active.insert(key); break }
            }
        }
        return active
    }

    /// Returns tile keys that need processing: active + neighbors with edge cells
    func tilesToProcess() -> [TileKey] {
        let active = activeTiles()
        var toProcess = active

        let T = tileSize
        for key in active {
            guard let buf = tiles[key] else { continue }
            let base = buf.baseAddress!

            // Top edge (row 0)
            for c in 0..<T {
                if isAlive(base[c]) {
                    toProcess.insert(TileKey(row: key.row - 1, col: key.col))
                    break
                }
            }
            // Bottom edge (row T-1)
            for c in 0..<T {
                if isAlive(base[(T - 1) * T + c]) {
                    toProcess.insert(TileKey(row: key.row + 1, col: key.col))
                    break
                }
            }
            // Left edge (col 0)
            for r in 0..<T {
                if isAlive(base[r * T]) {
                    toProcess.insert(TileKey(row: key.row, col: key.col - 1))
                    break
                }
            }
            // Right edge (col T-1)
            for r in 0..<T {
                if isAlive(base[r * T + T - 1]) {
                    toProcess.insert(TileKey(row: key.row, col: key.col + 1))
                    break
                }
            }
            // Corners
            if isAlive(base[0]) {
                toProcess.insert(TileKey(row: key.row - 1, col: key.col - 1))
            }
            if isAlive(base[T - 1]) {
                toProcess.insert(TileKey(row: key.row - 1, col: key.col + 1))
            }
            if isAlive(base[(T - 1) * T]) {
                toProcess.insert(TileKey(row: key.row + 1, col: key.col - 1))
            }
            if isAlive(base[(T - 1) * T + T - 1]) {
                toProcess.insert(TileKey(row: key.row + 1, col: key.col + 1))
            }
        }
        return Array(toProcess)
    }

    /// Build padded (T+2)×(T+2) tile into Float16 MLMultiArray
    func fillPadded(_ key: TileKey, into arr: MLMultiArray) {
        let T = tileSize
        let P = T + 2
        let dst = arr.dataPointer.assumingMemoryBound(to: UInt16.self)

        // Zero the whole padded tile
        memset(dst, 0, P * P * MemoryLayout<UInt16>.size)

        // Center: copy tile data into rows [1..T] cols [1..T]
        if let buf = tiles[key] {
            let src = buf.baseAddress!
            for r in 0..<T {
                memcpy(dst + (r + 1) * P + 1, src + r * T, T * MemoryLayout<UInt16>.size)
            }
        }

        // Top neighbor → row 0, cols [1..T]
        if let nb = tiles[TileKey(row: key.row - 1, col: key.col)] {
            let src = nb.baseAddress!
            memcpy(dst + 1, src + (T - 1) * T, T * MemoryLayout<UInt16>.size)
        }

        // Bottom neighbor → row T+1, cols [1..T]
        if let nb = tiles[TileKey(row: key.row + 1, col: key.col)] {
            let src = nb.baseAddress!
            memcpy(dst + (T + 1) * P + 1, src, T * MemoryLayout<UInt16>.size)
        }

        // Left neighbor → col 0, rows [1..T]
        if let nb = tiles[TileKey(row: key.row, col: key.col - 1)] {
            let src = nb.baseAddress!
            for r in 0..<T {
                dst[(r + 1) * P] = src[r * T + T - 1]
            }
        }

        // Right neighbor → col T+1, rows [1..T]
        if let nb = tiles[TileKey(row: key.row, col: key.col + 1)] {
            let src = nb.baseAddress!
            for r in 0..<T {
                dst[(r + 1) * P + T + 1] = src[r * T]
            }
        }

        // Corners
        if let nb = tiles[TileKey(row: key.row - 1, col: key.col - 1)] {
            dst[0] = nb.baseAddress![(T - 1) * T + T - 1]
        }
        if let nb = tiles[TileKey(row: key.row - 1, col: key.col + 1)] {
            dst[T + 1] = nb.baseAddress![(T - 1) * T]
        }
        if let nb = tiles[TileKey(row: key.row + 1, col: key.col - 1)] {
            dst[(T + 1) * P] = nb.baseAddress![T - 1]
        }
        if let nb = tiles[TileKey(row: key.row + 1, col: key.col + 1)] {
            dst[(T + 1) * P + T + 1] = nb.baseAddress![0]
        }
    }
}

// MARK: - Load Cells from Binary File

func loadCells(_ path: String, into tiled: TiledGOL) throws {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))

    guard data.count >= 24 else { throw NSError(domain: "GOL", code: 1, userInfo: [NSLocalizedDescriptionKey: "File too small"]) }
    guard data[0] == 0x47, data[1] == 0x4F, data[2] == 0x4C, data[3] == 0x00 else {
        throw NSError(domain: "GOL", code: 2, userInfo: [NSLocalizedDescriptionKey: "Bad magic"])
    }

    let nCells = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self) }
    let minRow = data.withUnsafeBytes { $0.load(fromByteOffset: 8, as: Int32.self) }
    let minCol = data.withUnsafeBytes { $0.load(fromByteOffset: 12, as: Int32.self) }
    let genLow = data.withUnsafeBytes { $0.load(fromByteOffset: 16, as: UInt32.self) }
    let genHigh = data.withUnsafeBytes { $0.load(fromByteOffset: 20, as: UInt32.self) }
    let gen = UInt64(genHigh) << 32 | UInt64(genLow)

    let ts = tiled.tileSize
    tiled.rowOffset = Int(minRow) >= 0 ? (Int(minRow) / ts) * ts : ((Int(minRow) - ts + 1) / ts) * ts
    tiled.colOffset = Int(minCol) >= 0 ? (Int(minCol) / ts) * ts : ((Int(minCol) - ts + 1) / ts) * ts
    tiled.generation = gen

    print("  Cells: \(nCells), gen: \(gen)")
    print("  Row offset: \(tiled.rowOffset), col offset: \(tiled.colOffset)")

    let headerSize = 24
    let expectedSize = headerSize + Int(nCells) * 8
    guard data.count >= expectedSize else {
        throw NSError(domain: "GOL", code: 3, userInfo: [NSLocalizedDescriptionKey: "Truncated"])
    }

    data.withUnsafeBytes { raw in
        let ptr = raw.baseAddress!.advanced(by: headerSize)
        for i in 0..<Int(nCells) {
            let r = Int(ptr.load(fromByteOffset: i * 8, as: Int32.self))
            let c = Int(ptr.load(fromByteOffset: i * 8 + 4, as: Int32.self))
            tiled.setCell(r, c)
        }
    }
}

// MARK: - CPU Reference Step

func cpuGOLStep(_ tiled: TiledGOL) {
    let T = tiled.tileSize
    let P = T + 2
    let keys = tiled.tilesToProcess()

    // Allocate scratch for padded tile (Float32 for CPU arithmetic)
    let padded = UnsafeMutablePointer<Float>.allocate(capacity: P * P)
    let result = UnsafeMutablePointer<UInt16>.allocate(capacity: T * T)
    defer {
        padded.deallocate()
        result.deallocate()
    }

    var newTiles: [TileKey: UnsafeMutableBufferPointer<UInt16>] = [:]

    for key in keys {
        memset(padded, 0, P * P * MemoryLayout<Float>.size)

        // Center - convert UInt16 (Float16) → Float32
        if let buf = tiled.tiles[key] {
            let src = buf.baseAddress!
            for r in 0..<T {
                for c in 0..<T {
                    padded[(r + 1) * P + c + 1] = isAlive(src[r * T + c]) ? 1.0 : 0.0
                }
            }
        }

        // Neighbors
        if let nb = tiled.tiles[TileKey(row: key.row - 1, col: key.col)] {
            let s = nb.baseAddress!
            for c in 0..<T { padded[1 + c] = isAlive(s[(T - 1) * T + c]) ? 1.0 : 0.0 }
        }
        if let nb = tiled.tiles[TileKey(row: key.row + 1, col: key.col)] {
            let s = nb.baseAddress!
            for c in 0..<T { padded[(T + 1) * P + 1 + c] = isAlive(s[c]) ? 1.0 : 0.0 }
        }
        if let nb = tiled.tiles[TileKey(row: key.row, col: key.col - 1)] {
            let s = nb.baseAddress!
            for r in 0..<T { padded[(r + 1) * P] = isAlive(s[r * T + T - 1]) ? 1.0 : 0.0 }
        }
        if let nb = tiled.tiles[TileKey(row: key.row, col: key.col + 1)] {
            let s = nb.baseAddress!
            for r in 0..<T { padded[(r + 1) * P + T + 1] = isAlive(s[r * T]) ? 1.0 : 0.0 }
        }
        if let nb = tiled.tiles[TileKey(row: key.row - 1, col: key.col - 1)] {
            padded[0] = isAlive(nb.baseAddress![(T - 1) * T + T - 1]) ? 1.0 : 0.0
        }
        if let nb = tiled.tiles[TileKey(row: key.row - 1, col: key.col + 1)] {
            padded[T + 1] = isAlive(nb.baseAddress![(T - 1) * T]) ? 1.0 : 0.0
        }
        if let nb = tiled.tiles[TileKey(row: key.row + 1, col: key.col - 1)] {
            padded[(T + 1) * P] = isAlive(nb.baseAddress![T - 1]) ? 1.0 : 0.0
        }
        if let nb = tiled.tiles[TileKey(row: key.row + 1, col: key.col + 1)] {
            padded[(T + 1) * P + T + 1] = isAlive(nb.baseAddress![0]) ? 1.0 : 0.0
        }

        // Apply B3/S23
        var hasAlive = false
        for r in 0..<T {
            for c in 0..<T {
                var n: Float = 0
                for dr in 0...2 {
                    for dc in 0...2 {
                        if dr == 1 && dc == 1 { continue }
                        n += padded[(r + dr) * P + c + dc]
                    }
                }
                let center = padded[(r + 1) * P + c + 1]
                let alive: Bool
                if center > 0.5 {
                    alive = (n == 2 || n == 3)
                } else {
                    alive = n == 3
                }
                result[r * T + c] = alive ? F16_ONE : F16_ZERO
                if alive { hasAlive = true }
            }
        }

        if hasAlive {
            let ptr = UnsafeMutablePointer<UInt16>.allocate(capacity: T * T)
            memcpy(ptr, result, T * T * MemoryLayout<UInt16>.size)
            newTiles[key] = UnsafeMutableBufferPointer(start: ptr, count: T * T)
        }
    }

    for (_, buf) in tiled.tiles {
        buf.baseAddress!.deallocate()
    }
    tiled.tiles = newTiles
    tiled.generation += 1
}

// MARK: - Reusable Feature Provider (avoids dictionary alloc per tile)

final class GridFeatureProvider: MLFeatureProvider {
    let gridValue: MLFeatureValue
    var featureNames: Set<String> { ["grid"] }

    init(array: MLMultiArray) {
        self.gridValue = MLFeatureValue(multiArray: array)
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        featureName == "grid" ? gridValue : nil
    }
}

// MARK: - ANE Simulation (Float16 native)

func aneSimulation(_ tiled: TiledGOL, model: MLModel, nGens: Int) throws {
    let T = tiled.tileSize
    let P = T + 2

    // Float16 input — no conversion needed, ANE runs natively in Float16
    let inputArr = try MLMultiArray(shape: [1, 1, NSNumber(value: P), NSNumber(value: P)],
                                     dataType: .float16)

    // Pre-allocate output array — avoids alloc per prediction, gives contiguous strides
    let outputArr = try MLMultiArray(shape: [1, 1, NSNumber(value: P), NSNumber(value: P)],
                                      dataType: .float16)
    let opts = MLPredictionOptions()
    opts.outputBackings = ["next_grid": outputArr]

    // Reusable feature provider — avoids dictionary alloc per tile
    let provider = GridFeatureProvider(array: inputArr)

    print("\nRunning \(nGens) generations on ANE...")

    for _ in 0..<nGens {
        let keys = tiled.tilesToProcess()
        var newTiles: [TileKey: UnsafeMutableBufferPointer<UInt16>] = [:]

        let t0 = CFAbsoluteTimeGetCurrent()

        for key in keys {
            // Fill padded tile directly in Float16
            tiled.fillPadded(key, into: inputArr)

            // Run model (reusing provider and output backing)
            let _ = try model.prediction(from: provider, options: opts)

            // Extract inner T×T from contiguous Float16 output (stride = P)
            let src = outputArr.dataPointer.assumingMemoryBound(to: UInt16.self)

            let ptr = UnsafeMutablePointer<UInt16>.allocate(capacity: T * T)
            var hasAlive = false

            for r in 0..<T {
                let srcRow = src + (r + 1) * P + 1
                let dstRow = ptr + r * T
                memcpy(dstRow, srcRow, T * MemoryLayout<UInt16>.size)
                if !hasAlive {
                    for c in 0..<T {
                        if isAlive(dstRow[c]) { hasAlive = true; break }
                    }
                }
            }

            if hasAlive {
                newTiles[key] = UnsafeMutableBufferPointer(start: ptr, count: T * T)
            } else {
                ptr.deallocate()
            }
        }

        // Free old tiles
        for (_, buf) in tiled.tiles {
            buf.baseAddress!.deallocate()
        }
        tiled.tiles = newTiles
        tiled.generation += 1

        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        let pop = tiled.population()
        print("  Gen \(tiled.generation): pop=\(pop), tiles=\(keys.count), " +
              "time=\(String(format: "%.2f", elapsed))s " +
              "(\(String(format: "%.1f", elapsed * 1000 / Double(max(keys.count, 1))))ms/tile)")
    }
}

// MARK: - Worker context for one compute engine

struct EngineWorker {
    let model: MLModel
    let inputArr: MLMultiArray
    let outputArr: MLMultiArray
    let opts: MLPredictionOptions
    let provider: GridFeatureProvider
    let label: String

    init(model: MLModel, tileSize: Int, label: String) throws {
        let P = tileSize + 2
        self.model = model
        self.label = label
        self.inputArr = try MLMultiArray(shape: [1, 1, NSNumber(value: P), NSNumber(value: P)],
                                          dataType: .float16)
        self.outputArr = try MLMultiArray(shape: [1, 1, NSNumber(value: P), NSNumber(value: P)],
                                           dataType: .float16)
        self.opts = MLPredictionOptions()
        self.opts.outputBackings = ["next_grid": outputArr]
        self.provider = GridFeatureProvider(array: inputArr)
    }
}

/// Process a slice of tiles on one engine, reading tile data from the shared TiledGOL.
/// Returns the new tiles computed by this engine.
func processTiles(_ keys: ArraySlice<TileKey>, tiled: TiledGOL,
                   worker: EngineWorker) -> [TileKey: UnsafeMutableBufferPointer<UInt16>] {
    let T = tiled.tileSize
    let P = T + 2
    var result: [TileKey: UnsafeMutableBufferPointer<UInt16>] = [:]

    for key in keys {
        // Fill padded tile — reads from tiled.tiles (immutable during this gen)
        tiled.fillPadded(key, into: worker.inputArr)

        // Predict
        let _ = try! worker.model.prediction(from: worker.provider, options: worker.opts)

        // Extract inner T×T
        let src = worker.outputArr.dataPointer.assumingMemoryBound(to: UInt16.self)
        let ptr = UnsafeMutablePointer<UInt16>.allocate(capacity: T * T)
        var hasAlive = false

        for r in 0..<T {
            let srcRow = src + (r + 1) * P + 1
            let dstRow = ptr + r * T
            memcpy(dstRow, srcRow, T * MemoryLayout<UInt16>.size)
            if !hasAlive {
                for c in 0..<T {
                    if isAlive(dstRow[c]) { hasAlive = true; break }
                }
            }
        }

        if hasAlive {
            result[key] = UnsafeMutableBufferPointer(start: ptr, count: T * T)
        } else {
            ptr.deallocate()
        }
    }
    return result
}

// MARK: - Dual-Engine Speculative Simulation (ANE + GPU in parallel)

func dualSimulation(_ tiled: TiledGOL, aneModel: MLModel, gpuModel: MLModel, nGens: Int) throws {
    let T = tiled.tileSize

    // Each engine gets its own input/output buffers (can't share across threads)
    let aneWorker = try EngineWorker(model: aneModel, tileSize: T, label: "ANE")
    let gpuWorker = try EngineWorker(model: gpuModel, tileSize: T, label: "GPU")

    // GPU is ~2× faster than ANE, so give it ~2/3 of the tiles
    let gpuFraction = 0.67

    print("\nRunning \(nGens) generations — speculative dual-engine (ANE + GPU)...")
    print("  Split: GPU ≈\(Int(gpuFraction * 100))%, ANE ≈\(Int((1.0 - gpuFraction) * 100))%")

    for _ in 0..<nGens {
        let keys = tiled.tilesToProcess()
        let splitIdx = Int(Double(keys.count) * gpuFraction)
        let gpuKeys = keys[0..<splitIdx]
        let aneKeys = keys[splitIdx...]

        let t0 = CFAbsoluteTimeGetCurrent()

        // Dispatch both engines in parallel
        var gpuTiles: [TileKey: UnsafeMutableBufferPointer<UInt16>] = [:]
        var aneTiles: [TileKey: UnsafeMutableBufferPointer<UInt16>] = [:]

        let group = DispatchGroup()

        group.enter()
        DispatchQueue.global(qos: .userInteractive).async {
            gpuTiles = processTiles(gpuKeys, tiled: tiled, worker: gpuWorker)
            group.leave()
        }

        group.enter()
        DispatchQueue.global(qos: .userInteractive).async {
            aneTiles = processTiles(aneKeys, tiled: tiled, worker: aneWorker)
            group.leave()
        }

        group.wait()

        let tPredict = CFAbsoluteTimeGetCurrent() - t0

        // Merge results from both engines
        var newTiles: [TileKey: UnsafeMutableBufferPointer<UInt16>] = gpuTiles
        for (k, v) in aneTiles { newTiles[k] = v }

        // Free old tiles
        for (_, buf) in tiled.tiles {
            buf.baseAddress!.deallocate()
        }
        tiled.tiles = newTiles
        tiled.generation += 1

        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        let pop = tiled.population()
        print("  Gen \(tiled.generation): pop=\(pop), tiles=\(keys.count) " +
              "(GPU:\(gpuKeys.count) ANE:\(aneKeys.count)), " +
              "predict=\(String(format: "%.2f", tPredict))s, " +
              "total=\(String(format: "%.2f", elapsed))s " +
              "(\(String(format: "%.1f", elapsed * 1000 / Double(max(keys.count, 1))))ms/tile)")
    }
}

// MARK: - Precompiled Padded-Tile Simulation (zero-copy, dual-engine)
//
// Tiles stored as P×P (padded) UInt16 buffers. Each generation:
//   1. Update only borders (~32KB/tile instead of 4.1MB fillPadded)
//   2. Zero-copy wrap as MLMultiArray — no memcpy into input
//   3. Output backing writes directly to new padded buffer — no extract copy
// Memory traffic: ~32KB/tile (was ~6.1MB/tile). 190× reduction.

func paddedUpdateBorders(key: TileKey, dst: UnsafeMutablePointer<UInt16>,
                          from tiles: [TileKey: UnsafeMutablePointer<UInt16>],
                          T: Int) {
    let P = T + 2
    // Zero border cells (top row, bottom row, left/right columns)
    memset(dst, 0, P * 2)
    memset(dst + (P - 1) * P, 0, P * 2)
    for r in 1...T { dst[r * P] = F16_ZERO; dst[r * P + P - 1] = F16_ZERO }

    // Top neighbor: its interior bottom row → our top border row
    if let n = tiles[TileKey(row: key.row - 1, col: key.col)] {
        memcpy(dst + 1, n + T * P + 1, T * 2)
    }
    // Bottom neighbor: its interior top row → our bottom border row
    if let n = tiles[TileKey(row: key.row + 1, col: key.col)] {
        memcpy(dst + (P - 1) * P + 1, n + P + 1, T * 2)
    }
    // Left neighbor: its interior right column → our left border column
    if let n = tiles[TileKey(row: key.row, col: key.col - 1)] {
        for r in 0..<T { dst[(r + 1) * P] = n[(r + 1) * P + T] }
    }
    // Right neighbor: its interior left column → our right border column
    if let n = tiles[TileKey(row: key.row, col: key.col + 1)] {
        for r in 0..<T { dst[(r + 1) * P + P - 1] = n[(r + 1) * P + 1] }
    }
    // Corners: neighbor interior corner cells
    if let n = tiles[TileKey(row: key.row - 1, col: key.col - 1)] { dst[0] = n[T * P + T] }
    if let n = tiles[TileKey(row: key.row - 1, col: key.col + 1)] { dst[P - 1] = n[T * P + 1] }
    if let n = tiles[TileKey(row: key.row + 1, col: key.col - 1)] { dst[(P - 1) * P] = n[P + T] }
    if let n = tiles[TileKey(row: key.row + 1, col: key.col + 1)] { dst[(P - 1) * P + P - 1] = n[P + 1] }
}

/// Check if a padded P×P tile has any alive cells in its interior
func paddedTileIsActive(_ p: UnsafeMutablePointer<UInt16>, T: Int) -> Bool {
    let P = T + 2
    for r in 0..<T {
        let row = p + (r + 1) * P + 1
        for c in 0..<T {
            if isAlive(row[c]) { return true }
        }
    }
    return false
}

/// Get tile keys to process from padded tile storage
func paddedTilesToProcess(_ tiles: [TileKey: UnsafeMutablePointer<UInt16>], T: Int) -> [TileKey] {
    let P = T + 2
    var active = Set<TileKey>()
    for (key, p) in tiles {
        if paddedTileIsActive(p, T: T) { active.insert(key) }
    }
    var toProcess = active
    for key in active {
        let p = tiles[key]!
        // Top edge of interior
        for c in 0..<T { if isAlive(p[P + 1 + c]) { toProcess.insert(TileKey(row: key.row - 1, col: key.col)); break } }
        // Bottom edge of interior
        for c in 0..<T { if isAlive(p[T * P + 1 + c]) { toProcess.insert(TileKey(row: key.row + 1, col: key.col)); break } }
        // Left edge of interior
        for r in 0..<T { if isAlive(p[(r + 1) * P + 1]) { toProcess.insert(TileKey(row: key.row, col: key.col - 1)); break } }
        // Right edge of interior
        for r in 0..<T { if isAlive(p[(r + 1) * P + T]) { toProcess.insert(TileKey(row: key.row, col: key.col + 1)); break } }
        // Corners
        if isAlive(p[P + 1]) { toProcess.insert(TileKey(row: key.row - 1, col: key.col - 1)) }
        if isAlive(p[P + T]) { toProcess.insert(TileKey(row: key.row - 1, col: key.col + 1)) }
        if isAlive(p[T * P + 1]) { toProcess.insert(TileKey(row: key.row + 1, col: key.col - 1)) }
        if isAlive(p[T * P + T]) { toProcess.insert(TileKey(row: key.row + 1, col: key.col + 1)) }
    }
    return Array(toProcess)
}

/// Population count from padded tiles
func paddedPopulation(_ tiles: [TileKey: UnsafeMutablePointer<UInt16>], T: Int) -> Int {
    let P = T + 2
    let ptrs = Array(tiles.values)
    let nThreads = min(8, ptrs.count)
    let chunk = (ptrs.count + nThreads - 1) / nThreads
    var partials = [Int](repeating: 0, count: nThreads)

    DispatchQueue.concurrentPerform(iterations: nThreads) { tIdx in
        let start = tIdx * chunk
        let end = min(start + chunk, ptrs.count)
        var count = 0
        for i in start..<end {
            let p = ptrs[i]
            for r in 0..<T {
                let row = p + (r + 1) * P + 1
                for c in 0..<T { if isAlive(row[c]) { count += 1 } }
            }
        }
        partials[tIdx] = count
    }
    return partials.reduce(0, +)
}

/// Process a slice of tiles with zero-copy MLMultiArray wrapping
func processTilesZeroCopy(_ keys: ArraySlice<TileKey>,
                            paddedTiles: [TileKey: UnsafeMutablePointer<UInt16>],
                            model: MLModel, T: Int) -> [TileKey: UnsafeMutablePointer<UInt16>] {
    let P = T + 2
    let PP = P * P
    let shape: [NSNumber] = [1, 1, NSNumber(value: P), NSNumber(value: P)]
    let strides: [NSNumber] = [NSNumber(value: PP), NSNumber(value: PP), NSNumber(value: P), 1]
    var result: [TileKey: UnsafeMutablePointer<UInt16>] = [:]
    let opts = MLPredictionOptions()

    for key in keys {
        autoreleasepool {
            guard let inputPtr = paddedTiles[key] else { return }

            // Zero-copy input: wrap existing padded buffer as MLMultiArray
            let inputArr = try! MLMultiArray(dataPointer: UnsafeMutableRawPointer(inputPtr),
                                              shape: shape, dataType: .float16,
                                              strides: strides, deallocator: nil)
            let provider = GridFeatureProvider(array: inputArr)

            // Output directly into new padded buffer
            let outPtr = UnsafeMutablePointer<UInt16>.allocate(capacity: PP)
            let outArr = try! MLMultiArray(dataPointer: UnsafeMutableRawPointer(outPtr),
                                            shape: shape, dataType: .float16,
                                            strides: strides, deallocator: nil)
            opts.outputBackings = ["next_grid": outArr]

            let _ = try! model.prediction(from: provider, options: opts)

            if paddedTileIsActive(outPtr, T: T) {
                result[key] = outPtr
            } else {
                outPtr.deallocate()
            }
        }
    }
    return result
}

func precompiledDualSimulation(_ tiled: TiledGOL, aneModel: MLModel, gpuModel: MLModel, nGens: Int) throws {
    let T = tiled.tileSize
    let P = T + 2
    let PP = P * P
    let gpuFraction = 0.67

    // Convert T×T tiles to P×P padded format
    print("  Converting to padded format...")
    let tConv0 = CFAbsoluteTimeGetCurrent()
    var padded: [TileKey: UnsafeMutablePointer<UInt16>] = [:]
    for (key, buf) in tiled.tiles {
        let p = UnsafeMutablePointer<UInt16>.allocate(capacity: PP)
        p.initialize(repeating: F16_ZERO, count: PP)
        let src = buf.baseAddress!
        for r in 0..<T {
            memcpy(p + (r + 1) * P + 1, src + r * T, T * 2)
        }
        padded[key] = p
    }
    let tConv = CFAbsoluteTimeGetCurrent() - tConv0
    let ramMB = Double(padded.count * PP * 2) / 1e6
    print("  \(padded.count) padded tiles (\(String(format: "%.0f", ramMB)) MB) in \(String(format: "%.2f", tConv))s")

    print("\nRunning \(nGens) generations — precompiled dual-engine (ANE + GPU, zero-copy)...")
    print("  Split: GPU ≈\(Int(gpuFraction * 100))%, ANE ≈\(Int((1.0 - gpuFraction) * 100))%")

    for _ in 0..<nGens {
        let keys = paddedTilesToProcess(padded, T: T)

        // Ensure all tiles to process have padded buffers
        for key in keys where padded[key] == nil {
            let p = UnsafeMutablePointer<UInt16>.allocate(capacity: PP)
            p.initialize(repeating: F16_ZERO, count: PP)
            padded[key] = p
        }

        let t0 = CFAbsoluteTimeGetCurrent()

        // Update borders (reads current gen, writes only border cells ~32KB/tile)
        for key in keys {
            paddedUpdateBorders(key: key, dst: padded[key]!, from: padded, T: T)
        }
        let tBorder = CFAbsoluteTimeGetCurrent() - t0

        // Split and dispatch to both engines
        let splitIdx = Int(Double(keys.count) * gpuFraction)
        let gpuKeys = keys[0..<splitIdx]
        let aneKeys = keys[splitIdx...]

        var gpuTiles: [TileKey: UnsafeMutablePointer<UInt16>] = [:]
        var aneTiles: [TileKey: UnsafeMutablePointer<UInt16>] = [:]

        let localPadded = padded  // snapshot for thread safety

        let group = DispatchGroup()
        group.enter()
        DispatchQueue.global(qos: .userInteractive).async {
            gpuTiles = processTilesZeroCopy(gpuKeys, paddedTiles: localPadded, model: gpuModel, T: T)
            group.leave()
        }
        group.enter()
        DispatchQueue.global(qos: .userInteractive).async {
            aneTiles = processTilesZeroCopy(aneKeys, paddedTiles: localPadded, model: aneModel, T: T)
            group.leave()
        }
        group.wait()

        let tPredict = CFAbsoluteTimeGetCurrent() - t0

        // Merge results
        var newPadded: [TileKey: UnsafeMutablePointer<UInt16>] = gpuTiles
        for (k, v) in aneTiles { newPadded[k] = v }

        // Free old padded tiles
        for (_, p) in padded { p.deallocate() }
        padded = newPadded
        tiled.generation += 1

        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        let pop = paddedPopulation(padded, T: T)
        print("  Gen \(tiled.generation): pop=\(pop), tiles=\(keys.count) " +
              "(GPU:\(gpuKeys.count) ANE:\(aneKeys.count)), " +
              "borders=\(String(format: "%.3f", tBorder))s, " +
              "predict=\(String(format: "%.2f", tPredict))s, " +
              "total=\(String(format: "%.2f", elapsed))s " +
              "(\(String(format: "%.1f", elapsed * 1000 / Double(max(keys.count, 1))))ms/tile)")
    }

    // Convert back to T×T tiles for final population
    for (_, buf) in tiled.tiles { buf.baseAddress!.deallocate() }
    tiled.tiles = [:]
    for (key, p) in padded {
        if paddedTileIsActive(p, T: T) {
            let ptr = UnsafeMutablePointer<UInt16>.allocate(capacity: T * T)
            for r in 0..<T {
                memcpy(ptr + r * T, p + (r + 1) * P + 1, T * 2)
            }
            tiled.tiles[key] = UnsafeMutableBufferPointer(start: ptr, count: T * T)
        }
        p.deallocate()
    }
}

// MARK: - mmap Arena for padded tiles

final class TileArena {
    let tileBytes: Int
    let maxTiles: Int
    private var base: UnsafeMutableRawPointer
    private var totalBytes: Int
    private var nextSlot: Int = 0

    init(tileSize: Int, maxTiles: Int) {
        let P = tileSize + 2
        self.tileBytes = P * P * MemoryLayout<UInt16>.size
        self.maxTiles = maxTiles
        self.totalBytes = tileBytes * maxTiles
        self.base = mmap(nil, totalBytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANON, -1, 0)!
    }

    deinit { munmap(base, totalBytes) }

    func alloc() -> UnsafeMutablePointer<UInt16> {
        let slot = nextSlot
        nextSlot += 1
        return (base + slot * tileBytes).assumingMemoryBound(to: UInt16.self)
    }

    func reset() { nextSlot = 0 }
}

// MARK: - Pipelined GPU Simulation (N-thread, mmap arena)

func pipelinedSimulation(_ tiled: TiledGOL, model: MLModel, nGens: Int, nWorkers: Int) throws {
    let T = tiled.tileSize
    let P = T + 2
    let PP = P * P

    // Two arenas: double-buffer between generations
    let arena0 = TileArena(tileSize: T, maxTiles: 1200)
    let arena1 = TileArena(tileSize: T, maxTiles: 1200)
    var arenas = [arena0, arena1]

    // Convert initial tiles to padded format in arena0
    var padded: [TileKey: UnsafeMutablePointer<UInt16>] = [:]
    for (key, buf) in tiled.tiles {
        let p = arena0.alloc()
        memset(p, 0, PP * 2)
        let src = buf.baseAddress!
        for r in 0..<T { memcpy(p + (r + 1) * P + 1, src + r * T, T * 2) }
        padded[key] = p
    }

    // Per-worker reusable MLMultiArray + provider
    struct Worker {
        let inputArr: MLMultiArray
        let outputArr: MLMultiArray
        let provider: GridFeatureProvider
        let opts: MLPredictionOptions
    }

    let shape: [NSNumber] = [1, 1, NSNumber(value: P), NSNumber(value: P)]
    var workers: [Worker] = []
    for _ in 0..<nWorkers {
        let inp = try MLMultiArray(shape: shape, dataType: .float16)
        let out = try MLMultiArray(shape: shape, dataType: .float16)
        let prov = GridFeatureProvider(array: inp)
        let opts = MLPredictionOptions()
        opts.outputBackings = ["next_grid": out]
        workers.append(Worker(inputArr: inp, outputArr: out, provider: prov, opts: opts))
    }

    // Warm up all workers
    for w in workers { let _ = try model.prediction(from: w.provider, options: w.opts) }

    let ramMB = Double(padded.count * PP * 2) / 1e6
    print("  \(padded.count) padded tiles (\(String(format: "%.0f", ramMB)) MB, mmap arena)")
    print("\nRunning \(nGens) generations — pipelined GPU (\(nWorkers) workers)...")

    var genIdx = 0
    for _ in 0..<nGens {
        let nxtArena = arenas[(genIdx + 1) % 2]
        nxtArena.reset()

        let keys = paddedTilesToProcess(padded, T: T)

        // Ensure all tiles to process have padded buffers
        let curArena = arenas[genIdx % 2]
        for key in keys where padded[key] == nil {
            let p = curArena.alloc()
            memset(p, 0, PP * 2)
            padded[key] = p
        }

        let t0 = CFAbsoluteTimeGetCurrent()

        // Border updates
        for key in keys {
            paddedUpdateBorders(key: key, dst: padded[key]!, from: padded, T: T)
        }
        let tBorder = CFAbsoluteTimeGetCurrent() - t0

        // Pipelined prediction with inline pop counting
        let chunkSize = (keys.count + nWorkers - 1) / nWorkers
        var newPadded: [TileKey: UnsafeMutablePointer<UInt16>] = [:]
        let lock = NSLock()
        var totalPop = 0

        let group = DispatchGroup()
        for wIdx in 0..<nWorkers {
            let start = wIdx * chunkSize
            let end = min(start + chunkSize, keys.count)
            if start >= end { continue }
            let slice = keys[start..<end]
            let w = workers[wIdx]

            group.enter()
            DispatchQueue.global(qos: .userInteractive).async {
                var localResult: [(TileKey, UnsafeMutablePointer<UInt16>)] = []
                var localPop = 0

                for key in slice {
                    memcpy(w.inputArr.dataPointer, padded[key]!, PP * 2)
                    let _ = try! model.prediction(from: w.provider, options: w.opts)

                    lock.lock()
                    let outPtr = nxtArena.alloc()
                    lock.unlock()
                    let src = w.outputArr.dataPointer.assumingMemoryBound(to: UInt16.self)
                    memcpy(outPtr, src, PP * 2)

                    // Count pop while output is hot in L1 cache
                    var tilePop = 0
                    for r in 0..<T {
                        let row = outPtr + (r + 1) * P + 1
                        for c in 0..<T { if isAlive(row[c]) { tilePop += 1 } }
                    }
                    localPop += tilePop
                    if tilePop > 0 {
                        localResult.append((key, outPtr))
                    }
                }

                lock.lock()
                for (k, p) in localResult { newPadded[k] = p }
                totalPop += localPop
                lock.unlock()
                group.leave()
            }
        }
        group.wait()

        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        padded = newPadded
        tiled.generation += 1
        genIdx += 1

        print("  Gen \(tiled.generation): pop=\(totalPop), tiles=\(keys.count), " +
              "borders=\(String(format: "%.0f", tBorder*1000))ms, " +
              "total=\(String(format: "%.0f", elapsed*1000))ms " +
              "(\(String(format: "%.0f", Double(keys.count) / elapsed)) tiles/s)")
    }

    // Convert back to T×T tiles
    for (_, buf) in tiled.tiles { buf.baseAddress!.deallocate() }
    tiled.tiles = [:]
    for (key, p) in padded {
        if paddedTileIsActive(p, T: T) {
            let ptr = UnsafeMutablePointer<UInt16>.allocate(capacity: T * T)
            for r in 0..<T { memcpy(ptr + r * T, p + (r + 1) * P + 1, T * 2) }
            tiled.tiles[key] = UnsafeMutableBufferPointer(start: ptr, count: T * T)
        }
    }
}

// MARK: - Main

func main() throws {
    let args = Array(CommandLine.arguments.dropFirst())
    var cpuOnly = false
    var dual = false
    var precompile = false
    var pipelineWorkers = 0
    var nGens = 3
    var cellsPath = "gol_computer.bin"
    var modelPath = "GOL_1026x1026_s1.mlmodelc"

    var i = 0
    while i < args.count {
        switch args[i] {
        case "--cpu-only": cpuOnly = true
        case "--dual": dual = true
        case "--precompile": precompile = true
        case "--pipeline": i += 1; pipelineWorkers = Int(args[i]) ?? 4
        case "--gens": i += 1; nGens = Int(args[i]) ?? 3
        case "--cells": i += 1; cellsPath = args[i]
        case "--model": i += 1; modelPath = args[i]
        default: break
        }
        i += 1
    }

    print(String(repeating: "=", count: 70))
    print("  GOL-on-ANE (Swift): Matrix Multiplication via Game of Life")
    print("  Chain: ANE (Conv2d) → GOL (B3/S23) → Computer → Matmul")
    print(String(repeating: "=", count: 70))
    fflush(stdout)
    print()

    // Load cells
    print("Loading cells from \(cellsPath)...")
    let tiled = TiledGOL(tileSize: TILE_SIZE)
    let loadStart = CFAbsoluteTimeGetCurrent()
    try loadCells(cellsPath, into: tiled)
    let loadElapsed = CFAbsoluteTimeGetCurrent() - loadStart
    let active = tiled.activeTiles().count
    print("  Loaded into \(active) tiles in \(String(format: "%.2f", loadElapsed))s")
    print("  Population: \(tiled.population())")
    print()

    if cpuOnly {
        print("Running \(nGens) generations on CPU...")
        for _ in 0..<nGens {
            let t0 = CFAbsoluteTimeGetCurrent()
            cpuGOLStep(tiled)
            let elapsed = CFAbsoluteTimeGetCurrent() - t0
            print("  Gen \(tiled.generation): pop=\(tiled.population()), " +
                  "time=\(String(format: "%.2f", elapsed))s")
        }
    } else {
        guard FileManager.default.fileExists(atPath: modelPath) else {
            print("Error: \(modelPath) not found.")
            print("Build it: python3 build_gol_model.py --height 1026 --width 1026")
            print("Compile:  xcrun coremlcompiler compile GOL_1026x1026_s1.mlpackage .")
            return
        }

        if pipelineWorkers > 0 {
            // Pipelined GPU: N workers feeding the same model concurrently
            print("Loading CoreML model (GPU): \(modelPath)")
            let configGPU = MLModelConfiguration()
            configGPU.computeUnits = .cpuAndGPU
            let gpuModel = try MLModel(contentsOf: URL(fileURLWithPath: modelPath), configuration: configGPU)
            print("  Loaded. Pipelined GPU mode.")
            print()

            try pipelinedSimulation(tiled, model: gpuModel, nGens: nGens, nWorkers: pipelineWorkers)
        } else if dual {
            // Speculative dual-engine: load model twice with different compute units
            print("Loading CoreML model (ANE): \(modelPath)")
            let configANE = MLModelConfiguration()
            configANE.computeUnits = .cpuAndNeuralEngine
            let aneModel = try MLModel(contentsOf: URL(fileURLWithPath: modelPath), configuration: configANE)

            print("Loading CoreML model (GPU): \(modelPath)")
            let configGPU = MLModelConfiguration()
            configGPU.computeUnits = .cpuAndGPU
            let gpuModel = try MLModel(contentsOf: URL(fileURLWithPath: modelPath), configuration: configGPU)

            print("  Both engines loaded. Speculative dual-dispatch mode.")
            print()

            if precompile {
                try precompiledDualSimulation(tiled, aneModel: aneModel, gpuModel: gpuModel, nGens: nGens)
            } else {
                try dualSimulation(tiled, aneModel: aneModel, gpuModel: gpuModel, nGens: nGens)
            }
        } else {
            print("Loading CoreML model: \(modelPath)")
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let model = try MLModel(contentsOf: URL(fileURLWithPath: modelPath), configuration: config)
            print("  Loaded. Compute units: all (CPU+GPU+ANE)")
            print()

            try aneSimulation(tiled, model: model, nGens: nGens)
        }
    }

    print("\nFinal state: gen=\(tiled.generation), pop=\(tiled.population())")
}

do {
    try main()
} catch {
    print("ERROR: \(error)")
    exit(1)
}
