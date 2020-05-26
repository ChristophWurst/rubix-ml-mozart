<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Layers;

use TEST_Tensor\TEST_Matrix;
use Test\Vendor\Rubix\ML\Deferred;
use Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Test\Vendor\Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;
use Test\Vendor\Rubix\ML\NeuralNet\ActivationFunctions\Softmax;
use Test\Vendor\Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss;
use InvalidArgumentException;
use RuntimeException;

use function count;

/**
 * Multiclass
 *
 * The Multiclass output layer gives a joint probability estimate of a multiclass classification
 * problem using the Softmax activation function.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Multiclass implements Output
{
    /**
     * The unique class labels.
     *
     * @var string[]
     */
    protected $classes = [
        //
    ];

    /**
     * The function that computes the loss of erroneous activations.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss
     */
    protected $costFn;

    /**
     * The softmax activation function.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\ActivationFunctions\Softmax
     */
    protected $activationFn;

    /**
     * The memorized input matrix.
     *
     * @var \TEST_Tensor\TEST_Matrix|null
     */
    protected $input;

    /**
     * The memorized activation matrix.
     *
     * @var \TEST_Tensor\TEST_Matrix|null
     */
    protected $computed;

    /**
     * @param string[] $classes
     * @param \Test\Vendor\Rubix\ML\NeuralNet\CostFunctions\ClassificationLoss|null $costFn
     * @throws \InvalidArgumentException
     */
    public function __construct(array $classes, ?ClassificationLoss $costFn = null)
    {
        $classes = array_values(array_unique($classes));

        if (count($classes) < 2) {
            throw new InvalidArgumentException('Number of classes'
                . ' must be greater than 1, ' . count($classes)
                . ' given.');
        }

        $this->classes = $classes;
        $this->costFn = $costFn ?? new CrossEntropy();
        $this->activationFn = new Softmax();
    }

    /**
     * Return the width of the layer.
     *
     * @return int
     */
    public function width() : int
    {
        return count($this->classes);
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
        $fanOut = count($this->classes);

        if ($fanIn !== $fanOut) {
            throw new InvalidArgumentException('Fan in must be'
                . " equal to fan out, $fanOut expected but"
                . " $fanIn given.");
        }

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
     * @throws \RuntimeException
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function infer(TEST_Matrix $input) : TEST_Matrix
    {
        return $this->activationFn->compute($input);
    }

    /**
     * Compute the gradient and loss at the output.
     *
     * @param string[] $labels
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @throws \RuntimeException
     * @return (\Test\Vendor\Rubix\ML\Deferred|float)[]
     */
    public function back(array $labels, Optimizer $optimizer) : array
    {
        if (!$this->input or !$this->computed) {
            throw new RuntimeException('Must perform forward pass'
                . ' before backpropagating.');
        }

        $expected = [];

        foreach ($this->classes as $class) {
            $dist = [];

            foreach ($labels as $label) {
                $dist[] = $class == $label ? 1.0 : 0.0;
            }

            $expected[] = $dist;
        }

        $expected = TEST_Matrix::quick($expected);

        $input = $this->input;
        $computed = $this->computed;

        $gradient = new Deferred([$this, 'gradient'], [$input, $computed, $expected]);

        $loss = $this->costFn->compute($computed, $expected);

        unset($this->input, $this->z, $this->computed);

        return [$gradient, $loss];
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @param \TEST_Tensor\TEST_Matrix $input
     * @param \TEST_Tensor\TEST_Matrix $computed
     * @param \TEST_Tensor\TEST_Matrix $expected
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function gradient(TEST_Matrix $input, TEST_Matrix $computed, TEST_Matrix $expected) : TEST_Matrix
    {
        if ($this->costFn instanceof CrossEntropy) {
            $dA = $computed->subtract($expected)
                ->divide($computed->n());
        }

        $dL = $this->costFn->differentiate($computed, $expected)
            ->divide($computed->n());

        return $this->activationFn->differentiate($input, $computed)
            ->multiply($dL);
    }
}
