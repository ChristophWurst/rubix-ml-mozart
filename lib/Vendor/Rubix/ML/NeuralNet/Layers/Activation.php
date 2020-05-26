<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Layers;

use TEST_Tensor\TEST_Matrix;
use Test\Vendor\Rubix\ML\Deferred;
use Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Test\Vendor\Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction;
use RuntimeException;

/**
 * Activation
 *
 * Activation layers apply a user-defined non-linear activation function to their
 * inputs.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Activation implements Hidden
{
    /**
     * The function that computes the output of the layer.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction
     */
    protected $activationFn;

    /**
     * The width of the layer.
     *
     * @var int|null
     */
    protected $width;

    /**
     * The memoized input matrix.
     *
     * @var \TEST_Tensor\TEST_Matrix|null
     */
    protected $input;

    /**
     * The memoized activation matrix.
     *
     * @var \TEST_Tensor\TEST_Matrix|null
     */
    protected $computed;

    /**
     * @param \Test\Vendor\Rubix\ML\NeuralNet\ActivationFunctions\ActivationFunction $activationFn
     */
    public function __construct(ActivationFunction $activationFn)
    {
        $this->activationFn = $activationFn;
    }

    /**
     * Return the width of the layer.
     *
     * @throws \RuntimeException
     * @return int
     */
    public function width() : int
    {
        if (!$this->width) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        return $this->width;
    }

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @param int $fanIn
     * @return int
     */
    public function initialize(int $fanIn) : int
    {
        $fanOut = $fanIn;

        $this->width = $fanOut;

        return $fanOut;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @param \TEST_Tensor\TEST_Matrix $input
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function forward(TEST_Matrix $input) : TEST_Matrix
    {
        $this->input = $input;

        $this->computed = $this->activationFn->compute($input);

        return $this->computed;
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param \TEST_Tensor\TEST_Matrix $input
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function infer(TEST_Matrix $input) : TEST_Matrix
    {
        return $this->activationFn->compute($input);
    }

    /**
     * Calculate the gradient and update the parameters of the layer.
     *
     * @param \Test\Vendor\Rubix\ML\Deferred $prevGradient
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @throws \RuntimeException
     * @return \Test\Vendor\Rubix\ML\Deferred
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred
    {
        if (!$this->input or !$this->computed) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $input = $this->input;
        $computed = $this->computed;

        unset($this->input, $this->computed);

        return new Deferred(
            [$this, 'gradient'],
            [$input, $computed, $prevGradient]
        );
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @param \TEST_Tensor\TEST_Matrix $input
     * @param \TEST_Tensor\TEST_Matrix $computed
     * @param \Test\Vendor\Rubix\ML\Deferred $prevGradient
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function gradient(TEST_Matrix $input, TEST_Matrix $computed, Deferred $prevGradient) : TEST_Matrix
    {
        return $this->activationFn->differentiate($input, $computed)
            ->multiply($prevGradient());
    }
}
