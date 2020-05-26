<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\ActivationFunctions;

use TEST_Tensor\TEST_Matrix;
use InvalidArgumentException;

/**
 * ELU
 *
 * Exponential Linear Units are a type of rectifier that soften the transition
 * from non-activated to activated using the exponential function.
 *
 * References:
 * [1] D. A. Clevert et al. (2016). Fast and Accurate Deep Network Learning by
 * Exponential Linear Units.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class ELU implements ActivationFunction
{
    /**
     * At which negative value the ELU will saturate. For example if alpha
     * equals 1, the leaked value will never be greater than -1.0.
     *
     * @var float
     */
    protected $alpha;

    /**
     * @param float $alpha
     * @throws \InvalidArgumentException
     */
    public function __construct(float $alpha = 1.0)
    {
        if ($alpha < 0.0) {
            throw new InvalidArgumentException('Alpha cannot be less than'
                . " 0, $alpha given.");
        }

        $this->alpha = $alpha;
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
        return $computed->map([$this, '_differentiate']);
    }

    /**
     * @param float $z
     * @return float
     */
    public function _compute(float $z) : float
    {
        return $z > 0.0 ? $z : $this->alpha * (exp($z) - 1.0);
    }

    /**
     * @param float $computed
     * @return float
     */
    public function _differentiate(float $computed) : float
    {
        return $computed > 0.0 ? 1.0 : $computed + $this->alpha;
    }
}
