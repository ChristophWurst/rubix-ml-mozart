<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\CostFunctions;

use TEST_Tensor\TEST_Matrix;

use const Test\Vendor\Rubix\ML\EPSILON;

/**
 * Relative Entropy
 *
 * Relative Entropy or *Kullback-Leibler divergence* is a measure of how the
 * expectation and activation of the network diverge.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class RelativeEntropy implements ClassificationLoss
{
    /**
     * Compute the loss.
     *
     * @param \TEST_Tensor\TEST_Matrix $output
     * @param \TEST_Tensor\TEST_Matrix $target
     * @return float
     */
    public function compute(TEST_Matrix $output, TEST_Matrix $target) : float
    {
        $target = $target->clip(EPSILON, 1.0);
        $output = $output->clip(EPSILON, 1.0);

        return $target->divide($output)->log()
            ->multiply($target)
            ->mean()
            ->mean();
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
        $target = $target->clip(EPSILON, 1.0);
        $output = $output->clip(EPSILON, 1.0);

        return $output->subtract($target)
            ->divide($output);
    }
}
