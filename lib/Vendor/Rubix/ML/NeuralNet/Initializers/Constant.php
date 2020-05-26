<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Initializers;

use TEST_Tensor\TEST_Matrix;
use InvalidArgumentException;

/**
 * Constant
 *
 * Initialize the parameter to a user specified constant value.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Constant implements Initializer
{
    /**
     * The value to initialize the parameter to.
     *
     * @var float
     */
    protected $value;

    /**
     * @param float $value
     * @throws \InvalidArgumentException
     */
    public function __construct(float $value = 0.0)
    {
        if (is_nan($value)) {
            throw new InvalidArgumentException('Cannot initialize'
                . ' weight values to NaN.');
        }

        $this->value = $value;
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
        return TEST_Matrix::fill($this->value, $fanOut, $fanIn);
    }
}
