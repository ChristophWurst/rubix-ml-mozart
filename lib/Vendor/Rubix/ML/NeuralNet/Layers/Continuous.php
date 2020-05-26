<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Layers;

use TEST_Tensor\TEST_Matrix;
use Test\Vendor\Rubix\ML\Deferred;
use Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Test\Vendor\Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Test\Vendor\Rubix\ML\NeuralNet\CostFunctions\RegressionLoss;
use InvalidArgumentException;
use RuntimeException;

/**
 * Continuous
 *
 * The Continuous output layer consists of a single linear neuron that outputs a scalar value.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Continuous implements Output
{
    /**
     * The function that computes the loss of erroneous activations.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\CostFunctions\RegressionLoss
     */
    protected $costFn;

    /**
     * The memorized input matrix.
     *
     * @var \TEST_Tensor\TEST_Matrix|null
     */
    protected $input;

    /**
     * @param \Test\Vendor\Rubix\ML\NeuralNet\CostFunctions\RegressionLoss|null $costFn
     */
    public function __construct(?RegressionLoss $costFn = null)
    {
        $this->costFn = $costFn ?? new LeastSquares();
    }

    /**
     * Return the width of the layer.
     *
     * @return int
     */
    public function width() : int
    {
        return 1;
    }

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @param int $fanIn
     * @throws \InvalidArgumentException
     * @return int
     */
    public function initialize(int $fanIn) : int
    {
        if ($fanIn !== 1) {
            throw new InvalidArgumentException('Fan in must be'
                . " equal to 1, $fanIn given.");
        }

        return 1;
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

        return $input;
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
     * Compute the gradient and loss at the output.
     *
     * @param (int|float)[] $labels
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @throws \RuntimeException
     * @return (\Test\Vendor\Rubix\ML\Deferred|float)[]
     */
    public function back(array $labels, Optimizer $optimizer) : array
    {
        if (!$this->input) {
            throw new RuntimeException('Must perform forward pass'
                . ' before backpropagating.');
        }

        $expected = TEST_Matrix::quick([$labels]);

        $input = $this->input;

        $gradient = new Deferred([$this, 'gradient'], [$input, $expected]);

        $loss = $this->costFn->compute($input, $expected);

        unset($this->input);

        return [$gradient, $loss];
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @param \TEST_Tensor\TEST_Matrix $input
     * @param \TEST_Tensor\TEST_Matrix $expected
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function gradient(TEST_Matrix $input, TEST_Matrix $expected) : TEST_Matrix
    {
        return $this->costFn->differentiate($input, $expected)
            ->divide($input->n());
    }
}
