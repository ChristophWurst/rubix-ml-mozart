<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\ActivationFunctions;

use TEST_Tensor\TEST_Matrix;

use const Test\Vendor\Rubix\ML\EPSILON;

/**
 * Softmax
 *
 * The Softmax function is a generalization of the Sigmoid function that squashes
 * each activation between 0 and 1, and all activations add up to 1.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Softmax extends Sigmoid
{
    /**
     * Compute the output value.
     *
     * @param \TEST_Tensor\TEST_Matrix $z
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function compute(TEST_Matrix $z) : TEST_Matrix
    {
        $zHat = $z->exp()->transpose();

        $total = $zHat->sum()->clipLower(EPSILON);

        return $zHat->divide($total)->transpose();
    }
}
