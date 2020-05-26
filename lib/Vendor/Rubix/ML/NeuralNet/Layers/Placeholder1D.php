<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Layers;

use TEST_Tensor\TEST_Matrix;
use InvalidArgumentException;

/**
 * Placeholder 1D
 *
 * The Placeholder 1D input layer represents the *future* input values of a mini
 * batch (matrix) of single dimensional tensors (vectors) to the neural network.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Placeholder1D implements Input
{
    /**
     * The number of input nodes. i.e. feature inputs.
     *
     * @var int
     */
    protected $inputs;

    /**
     * @param int $inputs
     * @throws \InvalidArgumentException
     */
    public function __construct(int $inputs)
    {
        if ($inputs < 1) {
            throw new InvalidArgumentException('Number of input nodes'
            . " must be greater than 0, $inputs given.");
        }

        $this->inputs = $inputs;
    }

    /**
     * @return int
     */
    public function width() : int
    {
        return $this->inputs;
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
        return $this->inputs;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @param \TEST_Tensor\TEST_Matrix $input
     * @throws \InvalidArgumentException
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function forward(TEST_Matrix $input) : TEST_Matrix
    {
        if ($input->m() !== $this->inputs) {
            throw new InvalidArgumentException('The number of features'
                . ' and input nodes must be equal,'
                . " $this->inputs expected but {$input->m()} given.");
        }

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
        return $this->forward($input);
    }
}
