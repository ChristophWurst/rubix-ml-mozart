<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Layers;

use TEST_Tensor\TEST_Matrix;
use Test\Vendor\Rubix\ML\Deferred;
use Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer;
use InvalidArgumentException;
use RuntimeException;

/**
 * Dropout
 *
 * Dropout is a regularization technique for reducing overfitting in neural
 * networks by preventing complex co-adaptations on training data. It works
 * by temporarily disabling neurons during each training pass. It also is a
 * very efficient way of performing model averaging with neural networks.
 *
 * References:
 * [1] N. Srivastava et al. (2014). Dropout: A Simple Way to Prevent Neural
 * Networks from Overfitting.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Dropout implements Hidden
{
    /**
     * The ratio of neurons that are dropped during each training pass.
     *
     * @var float
     */
    protected $ratio;

    /**
     * The scaling coefficient.
     *
     * @var float
     */
    protected $scale;

    /**
     * The width of the layer.
     *
     * @var int|null
     */
    protected $width;

    /**
     * The memoized dropout mask.
     *
     * @var \TEST_Tensor\TEST_Matrix|null
     */
    protected $mask;

    /**
     * @param float $ratio
     * @throws \InvalidArgumentException
     */
    public function __construct(float $ratio = 0.5)
    {
        if ($ratio <= 0.0 or $ratio >= 1.0) {
            throw new InvalidArgumentException('Ratio must be'
                . " between 0 and 1, $ratio given.");
        }

        $this->ratio = $ratio;
        $this->scale = 1.0 / (1.0 - $ratio);
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
        $this->mask = TEST_Matrix::rand(...$input->shape())
            ->greater($this->ratio)
            ->multiply($this->scale);

        return $this->mask->multiply($input);
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param \TEST_Tensor\TEST_Matrix $input
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function infer(TEST_Matrix $input) : TEST_Matrix
    {
        return $input;
    }

    /**
     * Calculate the gradients of the layer and update the parameters.
     *
     * @param \Test\Vendor\Rubix\ML\Deferred $prevGradient
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @throws \RuntimeException
     * @return \Test\Vendor\Rubix\ML\Deferred
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred
    {
        if (!$this->mask) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $mask = $this->mask;

        unset($this->mask);

        return new Deferred([$this, 'gradient'], [$prevGradient, $mask]);
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @param \Test\Vendor\Rubix\ML\Deferred $prevGradient
     * @param \TEST_Tensor\TEST_Matrix $mask
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function gradient(Deferred $prevGradient, TEST_Matrix $mask)
    {
        return $prevGradient()->multiply($mask);
    }
}
