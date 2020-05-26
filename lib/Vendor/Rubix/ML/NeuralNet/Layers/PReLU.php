<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Layers;

use TEST_Tensor\TEST_Matrix;
use Test\Vendor\Rubix\ML\Deferred;
use Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Test\Vendor\Rubix\ML\NeuralNet\Initializers\Constant;
use Test\Vendor\Rubix\ML\NeuralNet\Parameter;
use Test\Vendor\Rubix\ML\NeuralNet\Initializers\Initializer;
use RuntimeException;
use Generator;

/**
 * PReLU
 *
 * Parametric Rectified Linear Units are leaky rectifiers whose leakage coefficients
 * are learned during training.
 *
 * References:
 * [1] K. He et al. (2015). Delving Deep into Rectifiers: Surpassing Human-Level
 * Performance on ImageNet Classification.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class PReLU implements Hidden, Parametric
{
    /**
     * The initializer of the alpha (leakage) parameter.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\Initializers\Initializer
     */
    protected $initializer;

    /**
     * The width of the layer.
     *
     * @var int|null
     */
    protected $width;

    /**
     * The parameterized leakage coefficients.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\Parameter|null
     */
    protected $alpha;

    /**
     * The memoized input matrix.
     *
     * @var \TEST_Tensor\TEST_Matrix|null
     */
    protected $input;

    /**
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Initializers\Initializer|null $initializer
     */
    public function __construct(?Initializer $initializer = null)
    {
        $this->initializer = $initializer ?? new Constant(0.25);
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

        $alpha = $this->initializer->initialize(1, $fanOut)->columnAsVector(0);

        $this->alpha = new Parameter($alpha);

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

        return $this->compute($input);
    }

    /**
     * Compute an inferential pass through the layer.
     *
     * @param \TEST_Tensor\TEST_Matrix $input
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function infer(TEST_Matrix $input) : TEST_Matrix
    {
        return $this->compute($input);
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
        if (!$this->alpha) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        if (!$this->input) {
            throw new RuntimeException('Must perform forward pass'
                . ' before backpropagating.');
        }

        $dOut = $prevGradient();

        $dIn = $this->input->clipUpper(0.0);

        $dAlpha = $dOut->multiply($dIn)->sum();

        $this->alpha->update($optimizer->step($this->alpha, $dAlpha));

        $z = $this->input;

        unset($this->input);

        return new Deferred([$this, 'gradient'], [$z, $dOut]);
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @param \TEST_Tensor\TEST_Matrix $z
     * @param \TEST_Tensor\TEST_Matrix $dOut
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function gradient($z, $dOut) : TEST_Matrix
    {
        return $this->differentiate($z)->multiply($dOut);
    }

    /**
     * Compute the leaky ReLU activation function and return a matrix.
     *
     * @param \TEST_Tensor\TEST_Matrix $z
     * @throws \RuntimeException
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function compute(TEST_Matrix $z) : TEST_Matrix
    {
        if (!$this->alpha) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        $alphas = $this->alpha->param()->asArray();

        $computed = [];

        foreach ($z as $i => $row) {
            $alpha = $alphas[$i];

            $activations = [];

            foreach ($row as $value) {
                $activations[] = $value > 0.0
                    ? $value
                    : $alpha * $value;
            }

            $computed[] = $activations;
        }

        return TEST_Matrix::quick($computed);
    }

    /**
     * Calculate the derivative of the activation function at a given output.
     *
     * @param \TEST_Tensor\TEST_Matrix $z
     * @throws \RuntimeException
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function differentiate(TEST_Matrix $z) : TEST_Matrix
    {
        if (!$this->alpha) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        $alphas = $this->alpha->param()->asArray();

        $gradient = [];

        foreach ($z as $i => $row) {
            $leakage = $alphas[$i];

            $derivative = [];

            foreach ($row as $value) {
                $derivative[] = $value > 0.0 ? 1.0 : $leakage;
            }

            $gradient[] = $derivative;
        }

        return TEST_Matrix::quick($gradient);
    }

    /**
     * Return the parameters of the layer.
     *
     * @throws \RuntimeException
     * @return \Generator<\Test\Vendor\Rubix\ML\NeuralNet\Parameter>
     */
    public function parameters() : Generator
    {
        if (!$this->alpha) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        yield 'alpha' => $this->alpha;
    }

    /**
     * Restore the parameters in the layer from an associative array.
     *
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Parameter[] $parameters
     */
    public function restore(array $parameters) : void
    {
        $this->alpha = $parameters['alpha'];
    }
}
