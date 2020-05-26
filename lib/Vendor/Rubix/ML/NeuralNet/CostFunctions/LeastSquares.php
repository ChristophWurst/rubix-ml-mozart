<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\CostFunctions;

use TEST_Tensor\TEST_Matrix;

/**
 * Least Squares
 *
 * Least Squares or *quadratic* loss is a function that measures the squared
 * error between the target output and the actual output of a network.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class LeastSquares implements RegressionLoss
{
    /**
     * Compute the loss score.
     *
     * @param \TEST_Tensor\TEST_Matrix $output
     * @param \TEST_Tensor\TEST_Matrix $target
     * @return float
     */
    public function compute(TEST_Matrix $output, TEST_Matrix $target) : float
    {
        return $output->subtract($target)->square()->mean()->mean();
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @param \TEST_Tensor\TEST_Matrix $output
     * @param \TEST_Tensor\TEST_Matrix $target
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function differentiate(TEST_Matrix $output, TEST_Matrix $target) : TEST_Matrix
    {
        return $output->subtract($target);
    }
}
