<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\ActivationFunctions;

use TEST_Tensor\TEST_Matrix;
use InvalidArgumentException;

/**
 * Leaky ReLU
 *
 * Leaky Rectified Linear Units are functions that output x when x > 0 or a
 * small leakage value when x < 0. The amount of leakage is controlled by the
 * user-specified parameter.
 *
 * References:
 * [1] A. L. Maas et al. (2013). Rectifier Nonlinearities Improve Neural Network
 * Acoustic Models.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LeakyReLU implements ActivationFunction
{
    /**
     * The amount of leakage as a ratio of the input value to allow to pass
     * through when not activated.
     *
     * @var float
     */
    protected $leakage;

    /**
     * @param float $leakage
     * @throws \InvalidArgumentException
     */
    public function __construct(float $leakage = 0.1)
    {
        if ($leakage <= 0.0 or $leakage >= 1.0) {
            throw new InvalidArgumentException('Leakage must be between'
                . " 0 and 1, $leakage given.");
        }

        $this->leakage = $leakage;
    }

    /**
     * Compute the output value.
     *
     * @param \TEST_Tensor\TEST_Matrix $z
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function compute(TEST_Matrix $z) : TEST_Matrix
    {
        return $z->map([$this, '_compute']);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param \TEST_Tensor\TEST_Matrix $z
     * @param \TEST_Tensor\TEST_Matrix $computed
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function differentiate(TEST_Matrix $z, TEST_Matrix $computed) : TEST_Matrix
    {
        return $z->map([$this, '_differentiate']);
    }

    /**
     * @param float $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return $z > 0.0 ? $z : $this->leakage * $z;
    }

    /**
     * @param float $z
     * @return float
     */
    public function _differentiate(float $z) : float
    {
        return $z > 0.0 ? 1.0 : $this->leakage;
    }
}
