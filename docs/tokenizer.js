// Browser port of the Tokenizer in bpe_tokenizer.ipynb.
// The algorithm mirrors the notebook exactly so the demo and the Python
// implementation produce identical token ids.

export class BPETokenizer {
  constructor(vocab, merges) {
    this.vocab = vocab;
    this.merges = merges.map((m) => m.split(' '));
    this.tokenToId = new Map(vocab.map((t, i) => [t, i]));
  }

  /**
   * Split raw text into words and punctuation, lowercased.
   * Mirrors pretokenize() in the notebook, including its regex.
   */
  pretokenize(text) {
    // Python's \w is Unicode-aware (it matches é, ï, ñ); JavaScript's is
    // ASCII-only even under /u, which would split "café" into "caf" + "é".
    // \p{L}\p{N}\p{M}_ reproduces Python's class so both agree.
    const W = '[\\p{L}\\p{N}\\p{M}_]';
    const pattern = new RegExp(
      `${W}+(?:'[a-z]*)?|\\d+|[^\\s\\p{L}\\p{N}\\p{M}_]+|\\S`,
      'gu',
    );
    return text.toLowerCase().match(pattern) ?? [];
  }

  /**
   * Apply every merge rule in training order to one word.
   * Returns the final tokens plus the ordered list of merges that actually
   * fired, so the UI can replay how the word was built up.
   */
  mergeWord(word) {
    let tokens = Array.from(word + '</w>');
    const steps = [{ tokens: [...tokens], rule: null, index: -1 }];

    for (let r = 0; r < this.merges.length; r++) {
      const [a, b] = this.merges[r];
      const merged = a + b;

      let i = 0;
      let fired = false;
      const next = [];
      while (i < tokens.length) {
        if (i < tokens.length - 1 && tokens[i] === a && tokens[i + 1] === b) {
          next.push(merged);
          i += 2;
          fired = true;
        } else {
          next.push(tokens[i]);
          i += 1;
        }
      }
      tokens = next;
      if (fired) steps.push({ tokens: [...tokens], rule: [a, b], index: r });
    }

    return { tokens, steps };
  }

  /** Token ids for a piece of text, falling back to characters when unknown. */
  encode(text) {
    const ids = [];
    for (const word of this.pretokenize(text)) {
      for (const token of this.mergeWord(word).tokens) {
        if (this.tokenToId.has(token)) {
          ids.push(this.tokenToId.get(token));
        } else {
          for (const ch of token) {
            if (this.tokenToId.has(ch)) ids.push(this.tokenToId.get(ch));
          }
        }
      }
    }
    return ids;
  }

  /** Tokens with their ids, kept grouped per word for display. */
  encodeDetailed(text) {
    return this.pretokenize(text).map((word) => {
      const { tokens, steps } = this.mergeWord(word);
      const pieces = [];
      for (const token of tokens) {
        if (this.tokenToId.has(token)) {
          pieces.push({ token, id: this.tokenToId.get(token), known: true });
        } else {
          for (const ch of token) {
            if (this.tokenToId.has(ch)) {
              pieces.push({ token: ch, id: this.tokenToId.get(ch), known: false });
            }
          }
        }
      }
      return { word, pieces, steps };
    });
  }

  decode(ids) {
    return ids
      .map((id) => this.vocab[id] ?? '')
      .join('')
      .replaceAll('</w>', ' ')
      .trim();
  }
}
