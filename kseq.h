/* The MIT License

   Copyright (c) 2008, by Heng Li <lh3@sanger.ac.uk>

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

#ifndef AC_KSEQ_H
#define AC_KSEQ_H

#include <ctype.h>
#include <string.h>
#include <stdlib.h>

#ifndef KS_BUFSIZE
# define KS_BUFSIZE 4096
#endif

#ifndef KSTRING_T
#define KSTRING_T kstring_t
typedef struct __kstring_t {
	size_t l, m;
	char *s;
} kstring_t;
#endif

#ifndef kroundup32
#define kroundup32(x) (--(x), (x)|=(x)>>1, (x)|=(x)>>2, (x)|=(x)>>4, (x)|=(x)>>8, (x)|=(x)>>16, ++(x))
#endif

typedef struct {
	gzFile f;
	kstring_t name, comment, seq, qual;
} kseq_t;

static inline kseq_t *kseq_init(gzFile fd)
{
	kseq_t *s = (kseq_t*)calloc(1, sizeof(kseq_t));
	if (s) s->f = fd;
	return s;
}

/* caller should check that ks->f->end > 0*/
static inline void kseq_rewind(kseq_t *ks)
{
	gzrewind(ks->f);
}
static inline void kseq_destroy(kseq_t *ks)
{
	if (!ks) return;
	free(ks->name.s); free(ks->comment.s); free(ks->seq.s);	free(ks->qual.s);
	free(ks);
}

#define __realloc_or_ret(ksp)							\
	(ksp)->s = (char*)realloc((ksp)->s, (ksp)->m);				\
	if ((ksp)->s == NULL)							\
		return -4;

/* Return value:
   >=0  length of the sequence (normal)
   -1   end-of-file
   -2   instream read error
   -3   truncated quality string
   -4   allocation error
 */
static int kseq_read(kseq_t *seq)
{
	int c;
	if (seq->f == NULL)
		return -4;

	/* jump to the next header line */
	while ((c = gzgetc(seq->f)) != '>' && c != '@')
		if (c == -1) return -1; /* end of file, no string end */

	seq->name.l = seq->comment.l = seq->seq.l = seq->qual.l = 0;

	/* init name */
	kstring_t *ksp = &seq->name;
	ksp->m = KS_BUFSIZE;
	__realloc_or_ret(ksp)

	while((c = gzgetc(seq->f)) != '\n') {
		if (c == -1)
			goto end_of_file;

		/* omit spaces unless inside comments */
		if (isspace(c) == 0 || seq->comment.l) {
			ksp->s[ksp->l] = c;
			if (++ksp->l == ksp->m) {
				ksp->m <<= 1;
				__realloc_or_ret(ksp)
			}
		} else { /* end name, init comment */
			ksp->m = ksp->l + 1;
			__realloc_or_ret(ksp)
			ksp->s[ksp->l] = '\0';

			ksp = &seq->comment;
			ksp->m = KS_BUFSIZE;
			__realloc_or_ret(ksp)
		}
	}
	/* end seq->comment, if present, otherwise seq->name */
	ksp->m = ksp->l + 1;
	__realloc_or_ret(ksp)
	ksp->s[ksp->l] = '\0';

	/* init sequence */
	ksp = &seq->seq;
	ksp->m = KS_BUFSIZE;
	__realloc_or_ret(ksp)

	while (1) {
		if (seq->f->have > 0) {
			c = *seq->f->next;
		} else { /* can happen at last sequence of fasta */
			c = gzgetc(seq->f);
			/* mustn't return -1 at eof or we loose last sequence! */
			if (c < 0) break;
			/* This path break is unexpected. gzgetc() incremented the
			seq->f ->next, ->have and ->pos indices, so recover: */
			if (gzungetc(c, seq->f) < 0)
				goto end_of_file;
		}

		/* break before incrementing past first header character */
		/* so we can reread it */
		if (c == '>' || c == '@') break;

		++seq->f->next;
		--seq->f->have;
		++seq->f->pos;

		if (c == '+') break;

		if (isgraph(c)) {
			ksp->s[ksp->l] = (char)c;
			if (++ksp->l != ksp->m)
				continue;
			ksp->m <<= 1; /* double mem */
			__realloc_or_ret(ksp)
		}
	}

	/* end sequence */
	ksp->m = ksp->l + 1;
	__realloc_or_ret(ksp)

	if (c == '+') { /* FASTQ, skip the rest of '+' line */
		ksp->s[ksp->l] = '\0';
		do {
			c = gzgetc(seq->f);
			if (c == -1) {
				c = -3; /* we should not stop here */
				goto end_of_file;
			}
		} while (c != '\n');

		ksp = &seq->qual;
		ksp->m = seq->seq.m; /* NB: for 2bit we'll need ceiling(seq->seq.l * 4) */
		__realloc_or_ret(ksp)

		while (ksp->l < seq->seq.l) {
			c = gzgetc(seq->f);
			if (c == -1) {
				c = -3; /*lengths qual and sequence won't match */
				goto end_of_file;
			}
			if (c >= 33 && c <= 127) /* if out of rang no an error? */
				ksp->s[ksp->l++] = (unsigned char)c;
		}
		__realloc_or_ret(ksp)
	}
	c = seq->seq.l;
end_of_file:
	ksp->s[ksp->l] = '\0';
	return c;
}

#endif
