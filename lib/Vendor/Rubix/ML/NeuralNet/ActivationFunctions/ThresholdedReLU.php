<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\ActivationFunctions;

use TEST_Tensor\TEST_Matrix;
use InvalidArgumentException;

/**
 * Thresholded ReLU
 *
 * A Thresholded ReLU (Rectified Linear Unit) only outputs the signal above
 * a user-defined threshold parameter.
 *
 * References:
 * [1] K. Konda et al. (2015). Zero-bias Autoencoders and the Benefits of
 * Co-adapting Features.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ThresholdedReLU implements ActivationFunction
{
    /**
     * The input value necessary to trigger an activation.
     *
     * @var float
     */
    protected $threshold;

    /**
     * @param float $threshold
     * @throws \InvalidArgumentException
     */
    public function __construct(float $threshold = 1.)
    {
        if ($threshold < 0.0) {
            throw new InvalidArgumentException('Threshold must be'
                . " positive, $threshold given.");
        }

        $this->threshold = $threshold;
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
        return $z->greater($this->threshold);
    }

    /**
     * @param float $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return $z > $this->threshold ? $z : 0.0;
    }
}
