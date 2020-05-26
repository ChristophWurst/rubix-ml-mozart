<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Layers;

use TEST_Tensor\TEST_Matrix;
use Test\Vendor\Rubix\ML\Deferred;
use TEST_Tensor\TEST_ColumnVector;
use Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Test\Vendor\Rubix\ML\NeuralNet\Initializers\Constant;
use Test\Vendor\Rubix\ML\NeuralNet\Parameter;
use Test\Vendor\Rubix\ML\NeuralNet\Initializers\Initializer;
use InvalidArgumentException;
use RuntimeException;
use Generator;

use const Test\Vendor\Rubix\ML\EPSILON;

/**
 * Batch Norm
 *
 * Normalize the activations of the previous layer such that the mean activation
 * is close to 0 and the standard deviation is close to 1. Batch Norm can reduce
 * the amount of covariate shift within the network which makes it possible to use
 * higher learning rates and converge faster under some circumstances.
 *
 * References:
 * [1] S. Ioffe et al. (2015). Batch Normalization: Accelerating Deep Network
 * Training by Reducing Internal Covariate Shift.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class BatchNorm implements Hidden, Parametric
{
    /**
     * The decay rate of the previous running averages of the global mean
     * and variance.
     *
     * @var float
     */
    protected $decay;

    /**
     * The initializer for the beta parameter.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\Initializers\Initializer
     */
    protected $betaInitializer;

    /**
     * The initializer for the gamma parameter.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\Initializers\Initializer
     */
    protected $gammaInitializer;

    /**
     * The width of the layer. i.e. the number of neurons.
     *
     * @var int|null
     */
    protected $width;

    /**
     * The learnable centering parameter.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\Parameter|null
     */
    protected $beta;

    /**
     * The learnable scaling parameter.
     *
     * @var \Test\Vendor\Rubix\ML\NeuralNet\Parameter|null
     */
    protected $gamma;

    /**
     * The running mean of each input dimension.
     *
     * @var \TEST_Tensor\TEST_Vector|null
     */
    protected $mean;

    /**
     * The running variance of each input dimension.
     *
     * @var \TEST_Tensor\TEST_Vector|null
     */
    protected $variance;

    /**
     * A cache of inverse standard deviations calculated during the forward pass.
     *
     * @var \TEST_Tensor\TEST_Vector|null
     */
    protected $stdInv;

    /**
     * A cache of normalized inputs to the layer.
     *
     * @var \TEST_Tensor\TEST_Matrix|null
     */
    protected $xHat;

    /**
     * @param float $decay
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Initializers\Initializer|null $betaInitializer
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Initializers\Initializer|null $gammaInitializer
     * @throws \InvalidArgumentException
     */
    public function __construct(
        float $decay = 0.1,
        ?Initializer $betaInitializer = null,
        ?Initializer $gammaInitializer = null
    ) {
        if ($decay < 0.0 or $decay > 1.0) {
            throw new InvalidArgumentException('Decay must be'
                . " between 0 and 1, $decay given.");
        }

        $this->decay = $decay;
        $this->betaInitializer = $betaInitializer ?? new Constant(0.0);
        $this->gammaInitializer = $gammaInitializer ?? new Constant(1.0);
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

        $beta = $this->betaInitializer->initialize(1, $fanOut)->columnAsVector(0);
        $gamma = $this->gammaInitializer->initialize(1, $fanOut)->columnAsVector(0);

        $this->beta = new Parameter($beta);
        $this->gamma = new Parameter($gamma);

        $this->width = $fanOut;

        return $fanOut;
    }

    /**
     * Compute a forward pass through the layer.
     *
     * @param \TEST_Tensor\TEST_Matrix $input
     * @throws \RuntimeException
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function forward(TEST_Matrix $input) : TEST_Matrix
    {
        if (!$this->beta or !$this->gamma) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        $mean = $input->mean();
        $variance = $input->variance($mean)->clipLower(EPSILON);
        $stdInv = $variance->sqrt()->reciprocal();

        $xHat = $stdInv->multiply($input->subtract($mean));

        if (!$this->mean or !$this->variance) {
            $this->mean = $mean;
            $this->variance = $variance;
        }

        $this->mean = $this->mean->multiply(1.0 - $this->decay)
            ->add($mean->multiply($this->decay));

        $this->variance = $this->variance->multiply(1.0 - $this->decay)
            ->add($variance->multiply($this->decay));

        $this->stdInv = $stdInv;
        $this->xHat = $xHat;

        return $this->gamma->param()->multiply($xHat)
            ->add($this->beta->param());
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
        if (!$this->mean or !$this->variance or !$this->beta or !$this->gamma) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        $xHat = $input->subtract($this->mean)
            ->divide($this->variance->sqrt());

        return $this->gamma->param()->multiply($xHat)
            ->add($this->beta->param());
    }

    /**
     * Calculate the errors and gradients of the layer and update the parameters.
     *
     * @param \Test\Vendor\Rubix\ML\Deferred $prevGradient
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @throws \RuntimeException
     * @return \Test\Vendor\Rubix\ML\Deferred
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred
    {
        if (!$this->beta or !$this->gamma) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        if (!$this->stdInv or !$this->xHat) {
            throw new RuntimeException('Must perform forward pass before'
                . ' backpropagating.');
        }

        $dOut = $prevGradient();

        $dBeta = $dOut->sum();
        $dGamma = $dOut->multiply($this->xHat)->sum();

        $gamma = $this->gamma->param();

        $this->beta->update($optimizer->step($this->beta, $dBeta));
        $this->gamma->update($optimizer->step($this->gamma, $dGamma));

        $stdInv = $this->stdInv;
        $xHat = $this->xHat;

        unset($this->stdInv, $this->xHat);

        return new Deferred(
            [$this, 'gradient'],
            [$dOut, $gamma, $stdInv, $xHat]
        );
    }

    /**
     * Calculate the gradient for the previous layer.
     *
     * @param \TEST_Tensor\TEST_Matrix $dOut
     * @param \TEST_Tensor\TEST_ColumnVector $gamma
     * @param \TEST_Tensor\TEST_ColumnVector $stdInv
     * @param \TEST_Tensor\TEST_Matrix $xHat
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function gradient(TEST_Matrix $dOut, TEST_ColumnVector $gamma, TEST_ColumnVector $stdInv, TEST_Matrix $xHat) : TEST_Matrix
    {
        $dXHat = $dOut->multiply($gamma);

        $xHatSigma = $dXHat->multiply($xHat)->sum();

        $dXHatSigma = $dXHat->sum();

        return $dXHat->multiply($dOut->m())
            ->subtract($dXHatSigma)
            ->subtract($xHat->multiply($xHatSigma))
            ->multiply($stdInv->divide($dOut->m()));
    }

    /**
     * Return the parameters of the layer.
     *
     * @throws \RuntimeException
     * @return \Generator<\Test\Vendor\Rubix\ML\NeuralNet\Parameter>
     */
    public function parameters() : Generator
    {
        if (!$this->beta or !$this->gamma) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        yield 'beta' => $this->beta;
        yield 'gamma' => $this->gamma;
    }

    /**
     * Restore the parameters in the layer from an associative array.
     *
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Parameter[] $parameters
     */
    public function restore(array $parameters) : void
    {
        $this->beta = $parameters['beta'];
        $this->gamma = $parameters['gamma'];
    }
}
