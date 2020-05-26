<?php

namespace Test\Vendor\Rubix\ML\Transformers;

use TEST_Tensor\TEST_Matrix;
use Test\Vendor\Rubix\ML\DataType;
use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use InvalidArgumentException;
use RuntimeException;

/**
 * Gaussian Random Projector
 *
 * A Random Projector is a dimensionality reducer based on the
 * Johnson-Lindenstrauss lemma that uses a random matrix to project a feature
 * vector onto a user-specified number of dimensions. It is faster than most
 * non-randomized dimensionality reduction techniques and offers similar
 * performance. This version uses a random matrix sampled from a Gaussian
 * distribution.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class GaussianRandomProjector implements Transformer, Stateful
{
    /**
     * The target number of dimensions.
     *
     * @var int
     */
    protected $dimensions;

    /**
     * The random matrix.
     *
     * @var \TEST_Tensor\TEST_Matrix|null
     */
    protected $r;

    /**
     * Calculate the minimum number of dimensions for n total samples with a
     * given maximum distortion using the Johnson-Lindenstrauss lemma.
     *
     * @param int $n
     * @param float $maxDistortion
     * @return int
     */
    public static function minDimensions(int $n, float $maxDistortion = 0.1) : int
    {
        $denominator = $maxDistortion ** 2 / 2.0 - $maxDistortion ** 3 / 3.0;

        return (int) round(4.0 * log($n) / $denominator);
    }

    /**
     * @param int $dimensions
     * @throws \InvalidArgumentException
     */
    public function __construct(int $dimensions)
    {
        if ($dimensions < 1) {
            throw new InvalidArgumentException('Dimensions must be'
                . " greater than 0, $dimensions given.");
        }

        $this->dimensions = $dimensions;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return \Test\Vendor\Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->r);
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::check($dataset, $this);

        $this->r = TEST_Matrix::gaussian($dataset->numColumns(), $this->dimensions);
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (!$this->r) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        $samples = TEST_Matrix::quick($samples)
            ->matmul($this->r)
            ->asArray();
    }
}
