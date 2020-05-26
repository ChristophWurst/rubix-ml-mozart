<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Initializers;

use TEST_Tensor\TEST_Matrix;
use InvalidArgumentException;

/**
 * Normal
 *
 * Generates a random weight matrix from a Gaussian distribution with
 * user-specified standard deviation.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Normal implements Initializer
{
    /**
     * The standard deviation of the distribution to sample from.
     *
     * @var float
     */
    protected $stddev;

    /**
     * @param float $stddev
     * @throws \InvalidArgumentException
     */
    public function __construct(float $stddev = 0.05)
    {
        if ($stddev <= 0.0) {
            throw new InvalidArgumentException('Standard deviation must'
                . " be greater than 0, $stddev given.");
        }

        $this->stddev = $stddev;
    }

    /**
     * Initialize a weight matrix W in the dimensions fan in x fan out.
     *
     * @param int $fanIn
     * @param int $fanOut
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function initialize(int $fanIn, int $fanOut) : TEST_Matrix
    {
        return TEST_Matrix::gaussian($fanOut, $fanIn)->multiply($this->stddev);
    }
}
