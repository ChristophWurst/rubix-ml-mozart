<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\CostFunctions;

use TEST_Tensor\TEST_Matrix;

use const Test\Vendor\Rubix\ML\EPSILON;

/**
 * Cross Entropy
 *
 * Cross Entropy, or log loss, measures the performance of a classification model
 * whose output is a probability value between 0 and 1. Cross-entropy loss
 * increases as the predicted probability diverges from the actual label. So
 * predicting a probability of .012 when the actual observation label is 1 would
 * be bad and result in a high loss value. A perfect score would have a log loss
 * of 0.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class CrossEntropy implements ClassificationLoss
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
        $entropy = $output->clipLower(EPSILON)->log();

        return $target->negate()->multiply($entropy)->mean()->mean();
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
        $denominator = TEST_Matrix::ones(...$target->shape())
            ->subtract($output)
            ->multiply($output)
            ->clipLower(EPSILON);

        return $output->subtract($target)
            ->divide($denominator);
    }
}
