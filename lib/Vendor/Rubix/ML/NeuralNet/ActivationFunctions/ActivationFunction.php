<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\ActivationFunctions;

use TEST_Tensor\TEST_Matrix;

interface ActivationFunction
{
    /**
     * Compute the output value.
     *
     * @param \TEST_Tensor\TEST_Matrix $z
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function compute(TEST_Matrix $z) : TEST_Matrix;

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param \TEST_Tensor\TEST_Matrix $z
     * @param \TEST_Tensor\TEST_Matrix $computed
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function differentiate(TEST_Matrix $z, TEST_Matrix $computed) : TEST_Matrix;
}
