<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\CostFunctions;

use TEST_Tensor\TEST_Matrix;

interface CostFunction
{
    /**
     * Compute the loss score.
     *
     * @param \TEST_Tensor\TEST_Matrix $output
     * @param \TEST_Tensor\TEST_Matrix $target
     * @return float
     */
    public function compute(TEST_Matrix $output, TEST_Matrix $target) : float;

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @param \TEST_Tensor\TEST_Matrix $output
     * @param \TEST_Tensor\TEST_Matrix $target
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function differentiate(TEST_Matrix $output, TEST_Matrix $target) : TEST_Matrix;
}
