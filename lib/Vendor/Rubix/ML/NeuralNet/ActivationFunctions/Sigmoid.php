<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\ActivationFunctions;

use TEST_Tensor\TEST_Matrix;

/**
 * Sigmoid
 *
 * A bounded S-shaped function (sometimes called the *Logistic* function) with an output value
 * between 0 and 1. The output of the sigmoid function has the advantage of being interpretable
 * as a probability, however it is not zero-centered and tends to saturate if inputs become large.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Sigmoid implements ActivationFunction
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
        return $computed->map([$this, '_differentiate']);
    }

    /**
     * @param float $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return 1.0 / (1.0 + exp(-$z));
    }

    /**
     * @param float $computed
     * @return float
     */
    public function _differentiate(float $computed) : float
    {
        return $computed * (1.0 - $computed);
    }
}
