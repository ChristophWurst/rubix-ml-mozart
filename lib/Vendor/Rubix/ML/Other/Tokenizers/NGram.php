<?php

namespace Test\Vendor\Rubix\ML\Other\Tokenizers;

use InvalidArgumentException;

use function count;

/**
 * N-gram
 *
 * N-grams are sequences of n-words of a given string. The N-gram tokenizer
 * outputs tokens of contiguous words ranging from *min* to *max* number of
 * words per token.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NGram implements Tokenizer
{
    /**
     * The regular expression to match sentences in a blob of text.
     *
     * @var string
     */
    protected const SENTENCE_REGEX = '/(?<=[.?!])\s+(?=[a-z])/i';

    /**
     * The separator between words in the n-gram.
     *
     * @var string
     */
    protected const SEPARATOR = ' ';

    /**
     * The minimum number of contiguous words in a single token.
     *
     * @var int
     */
    protected $min;

    /**
     * The maximum number of contiguous words in a single token.
     *
     * @var int
     */
    protected $max;

    /**
     * The word tokenizer.
     *
     * @var \Test\Vendor\Rubix\ML\Other\Tokenizers\Word
     */
    protected $wordTokenizer;

    /**
     * @param int $min
     * @param int $max
     * @param \Test\Vendor\Rubix\ML\Other\Tokenizers\Word|null $wordTokenizer
     * @throws \InvalidArgumentException
     */
    public function __construct(int $min = 2, int $max = 2, Word $wordTokenizer = null)
    {
        if ($min < 1) {
            throw new InvalidArgumentException('Minimum cannot be less'
                . ' than 1.');
        }

        if ($max < $min) {
            throw new InvalidArgumentException('Maximum cannot be less'
                . ' than minimum.');
        }

        $this->min = $min;
        $this->max = $max;
        $this->wordTokenizer = $wordTokenizer ?? new Word();
    }

    /**
     * Tokenize a block of text.
     *
     * @param string $string
     * @return string[]
     */
    public function tokenize(string $string) : array
    {
        $sentences = preg_split(self::SENTENCE_REGEX, $string) ?: [];

        $tokens = [];

        foreach ($sentences as $sentence) {
            $words = $this->wordTokenizer->tokenize($sentence);

            $n = count($words);

            foreach ($words as $i => $word) {
                $p = min($n - $i, $this->max);

                for ($j = $this->min; $j <= $p; ++$j) {
                    $nGram = $word;

                    for ($k = 1; $k < $j; ++$k) {
                        $nGram .= self::SEPARATOR . $words[$i + $k];
                    }

                    $tokens[] = $nGram;
                }
            }
        }

        return $tokens;
    }
}
