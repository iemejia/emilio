/*
 * eml_mov.h -- Compatibility shim for eml_tokenizer.c
 *
 * The tokenizer source (#include "eml_mov.h") was written for the MOV engine.
 * This shim re-exports the identical type definitions from eml_tag.h so that
 * eml_tokenizer.c compiles unmodified against the tag-system engine.
 */
#ifndef EML_MOV_H
#define EML_MOV_H
#include "eml_tag.h"
#endif /* EML_MOV_H */
