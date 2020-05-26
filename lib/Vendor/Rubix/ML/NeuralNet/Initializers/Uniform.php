<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Initializers;

use TEST_Tensor\TEST_Matrix;
use InvalidArgumentException;

/**
 * Uniform
 *
 * Generates a random uniform distribution centered at 0 and bounded at
 * both ends by the parameter beta.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Uniform implements Initializer
{
    /**
     * The upper and lower bound of the distribution.
     *
     * @var float
     */
    protected $beta;

    /**
     * @param float $beta
     * @throws \InvalidArgumentException
     */
    public function __construct(float $beta = 0.5)
    {
        if ($beta <= 0.0) {
            throw new InvalidArgumentException('Beta cannot be less than'
                . " or equal to 0, $beta given.");
        }

        $this->beta = $beta;
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
        return TEST_Matrix::uniform($fanOut, $fanIn)->multiply($this->beta);
    }
}
