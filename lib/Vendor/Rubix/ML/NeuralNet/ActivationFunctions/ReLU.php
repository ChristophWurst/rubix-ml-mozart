<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\ActivationFunctions;

use TEST_Tensor\TEST_Matrix;

/**
 * ReLU
 *
 * ReLU (Rectified Linear Unit) is an activation function that only outputs
 * the positive signal of the input.
 *
 * References:
 * [1] A. L. Maas et al. (2013). Rectifier Nonlinearities Improve Neural
 * Network Acoustic Models.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ReLU implements ActivationFunction
{
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
        return $z->greater(0.0);
    }

    /**
     * @param float $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return $z > 0.0 ? $z : 0.0;
    }
}
