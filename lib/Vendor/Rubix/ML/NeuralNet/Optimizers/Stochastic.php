<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Optimizers;

use TEST_Tensor\Tensor;
use Test\Vendor\Rubix\ML\NeuralNet\Parameter;
use InvalidArgumentException;

/**
 * Stochastic
 *
 * A constant learning rate gradient descent optimizer.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Stochastic implements Optimizer
{
    /**
     * The learning rate that controls the global step size.
     *
     * @var float
     */
    protected $rate;

    /**
     * @param float $rate
     * @throws \InvalidArgumentException
     */
    public function __construct(float $rate = 0.01)
    {
        if ($rate <= 0.0) {
            throw new InvalidArgumentException('Learning rate must'
                . " be greater than 0, $rate given.");
        }

        $this->rate = $rate;
    }

    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Parameter $param
     * @param \TEST_Tensor\Tensor<int|float|array> $gradient
     * @return \TEST_Tensor\Tensor<int|float|array>
     */
    public function step(Parameter $param, TEST_Tensor $gradient) : TEST_Tensor
    {
        return $gradient->multiply($this->rate);
    }
}
