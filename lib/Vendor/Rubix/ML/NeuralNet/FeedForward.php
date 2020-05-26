<?php

namespace Test\Vendor\Rubix\ML\NeuralNet;

use TEST_Tensor\TEST_Matrix;
use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Datasets\Labeled;
use Test\Vendor\Rubix\ML\NeuralNet\Layers\Layer;
use Test\Vendor\Rubix\ML\NeuralNet\Layers\Input;
use Test\Vendor\Rubix\ML\NeuralNet\Layers\Hidden;
use Test\Vendor\Rubix\ML\NeuralNet\Layers\Output;
use Test\Vendor\Rubix\ML\NeuralNet\Layers\Parametric;
use Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Adaptive;
use Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer;
use InvalidArgumentException;
use Traversable;

/**
 * Feed Forward
 *
 * A feed forward neural network implementation consisting of an input and
 * output layer and any number of intermediate hidden layers.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class FeedForward implements Network
{
    /**
     * The input layer to the network.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\Layers\Input
     */
    protected $input;

    /**
     * The hidden layers of the network.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\Layers\Hidden[]
     */
    protected $hidden = [
        //
    ];

    /**
     * The pathing of the backward pass through the hidden layers.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\Layers\Hidden[]
     */
    protected $backPass = [
        //
    ];

    /**
     * The output layer of the network.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\Layers\Output
     */
    protected $output;

    /**
     * The gradient descent optimizer used to train the network.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Layers\Input $input
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Layers\Hidden[] $hidden
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Layers\Output $output
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     */
    public function __construct(Input $input, array $hidden, Output $output, Optimizer $optimizer)
    {
        foreach ($hidden as $layer) {
            if (!$layer instanceof Hidden) {
                throw new InvalidArgumentException('Hidden layer must'
                    . ' implement the Hidden interface.');
            }
        }

        $layers = [$input];
        $layers = array_merge($layers, $hidden);
        $layers[] = $output;

        foreach ($layers as $layer) {
            $fanIn = $layer->initialize($fanIn ?? 0);
        }

        if ($optimizer instanceof Adaptive) {
            foreach ($layers as $layer) {
                if ($layer instanceof Parametric) {
                    foreach ($layer->parameters() as $param) {
                        $optimizer->warm($param);
                    }
                }
            }
        }

        $this->input = $input;
        $this->hidden = $hidden;
        $this->output = $output;
        $this->backPass = array_reverse($hidden);
        $this->optimizer = $optimizer;
    }

    /**
     * Return the input layer.
     *
     * @return \Test\Vendor\Rubix\ML\NeuralNet\Layers\Input
     */
    public function input() : Input
    {
        return $this->input;
    }

    /**
     * Return an array of hidden layers indexed left to right.
     *
     * @return \Test\Vendor\Rubix\ML\NeuralNet\Layers\Hidden[]
     */
    public function hidden() : array
    {
        return $this->hidden;
    }

    /**
     * Return the output layer.
     *
     * @return \Test\Vendor\Rubix\ML\NeuralNet\Layers\Output
     */
    public function output() : Output
    {
        return $this->output;
    }

    /**
     * Return all the layers in the network.
     *
     * @return \Traversable<\Test\Vendor\Rubix\ML\NeuralNet\Layers\Layer>
     */
    public function layers() : Traversable
    {
        yield $this->input;
        
        foreach ($this->hidden as $hidden) {
            yield $hidden;
        }

        yield $this->output;
    }

    /**
     * Run an inference pass and return the activations at the output layer.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function infer(Dataset $dataset) : TEST_Matrix
    {
        $input = TEST_Matrix::quick($dataset->samples())->transpose();

        $input = $this->input->infer($input);

        foreach ($this->hidden as $hidden) {
            $input = $hidden->infer($input);
        }

        $activations = $this->output->infer($input)->transpose();

        return $activations;
    }

    /**
     * Perform a forward and backward pass of the network in one call. Returns
     * the loss from the backward pass.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Labeled $dataset
     * @return float
     */
    public function roundtrip(Labeled $dataset) : float
    {
        $input = TEST_Matrix::quick($dataset->samples())->transpose();

        $this->feed($input);
        
        $loss = $this->backpropagate($dataset->labels());

        return $loss;
    }

    /**
     * Feed a batch through the network and return a matrix of activations.
     *
     * @param \TEST_Tensor\TEST_Matrix $input
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function feed(TEST_Matrix $input) : TEST_Matrix
    {
        $input = $this->input->forward($input);

        foreach ($this->hidden as $hidden) {
            $input = $hidden->forward($input);
        }

        return $this->output->forward($input);
    }

    /**
     * Backpropagate the gradient produced by the cost function and return the loss.
     *
     * @param (string|int|float)[] $labels
     * @return float
     */
    public function backpropagate(array $labels) : float
    {
        [$gradient, $loss] = $this->output->back($labels, $this->optimizer);

        foreach ($this->backPass as $layer) {
            $gradient = $layer->back($gradient, $this->optimizer);
        }

        return $loss;
    }
}
