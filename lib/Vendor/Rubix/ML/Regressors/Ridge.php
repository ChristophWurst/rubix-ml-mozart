<?php

namespace Test\Vendor\Rubix\ML\Regressors;

use TEST_Tensor\TEST_Matrix;
use TEST_Tensor\TEST_Vector;
use Test\Vendor\Rubix\ML\Learner;
use Test\Vendor\Rubix\ML\DataType;
use Test\Vendor\Rubix\ML\Estimator;
use Test\Vendor\Rubix\ML\Persistable;
use Test\Vendor\Rubix\ML\RanksFeatures;
use Test\Vendor\Rubix\ML\EstimatorType;
use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Datasets\Labeled;
use Test\Vendor\Rubix\ML\Other\Traits\PredictsSingle;
use Test\Vendor\Rubix\ML\Specifications\DatasetIsNotEmpty;
use Test\Vendor\Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use Test\Vendor\Rubix\ML\Specifications\SamplesAreCompatibleWithEstimator;
use InvalidArgumentException;
use RuntimeException;

use function is_null;

/**
 * Ridge
 *
 * L2 regularized least squares linear model solved using a closed-form solution. The addition
 * of regularization, controlled by the *alpha* parameter, makes Ridge less prone to overfitting
 * than ordinary linear regression.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Ridge implements Estimator, Learner, RanksFeatures, Persistable
{
    use PredictsSingle;
    
    /**
     * The strength of the L2 regularization penalty.
     *
     * @var float
     */
    protected $alpha;

    /**
     * The y intercept i.e. the bias added to the decision function.
     *
     * @var float|null
     */
    protected $bias;

    /**
     * The computed coefficients of the regression line.
     *
     * @var \TEST_Tensor\TEST_Vector|null
     */
    protected $coefficients;

    /**
     * @param float $alpha
     * @throws \InvalidArgumentException
     */
    public function __construct(float $alpha = 1.0)
    {
        if ($alpha < 0.0) {
            throw new InvalidArgumentException('Alpha must be'
                . " greater than 0, $alpha given.");
        }

        $this->alpha = $alpha;
    }

    /**
     * Return the estimator type.
     *
     * @return \Test\Vendor\Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return EstimatorType::regressor();
    }

    /**
     * Return the data types that the model is compatible with.
     *
     * @return \Test\Vendor\Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return [
            DataType::continuous(),
        ];
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'alpha' => $this->alpha,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->coefficients and isset($this->bias);
    }

    /**
     * Return the weights of features in the decision function.
     *
     * @return (int|float)[]|null
     */
    public function coefficients() : ?array
    {
        return $this->coefficients ? $this->coefficients->asArray() : null;
    }

    /**
     * Return the bias added to the decision function.
     *
     * @return float|null
     */
    public function bias() : ?float
    {
        return $this->bias;
    }

    /**
     * Train the learner with a dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function train(Dataset $dataset) : void
    {
        if (!$dataset instanceof Labeled) {
            throw new InvalidArgumentException('Learner requires a'
                . ' Labeled training set.');
        }

        DatasetIsNotEmpty::check($dataset);
        SamplesAreCompatibleWithEstimator::check($dataset, $this);
        LabelsAreCompatibleWithLearner::check($dataset, $this);

        $biases = TEST_Matrix::ones($dataset->numRows(), 1);

        $x = TEST_Matrix::build($dataset->samples())->augmentLeft($biases);
        $y = TEST_Vector::build($dataset->labels());

        $alphas = array_fill(0, $x->n() - 1, $this->alpha);

        array_unshift($alphas, 0.0);

        $penalties = TEST_Matrix::diagonal($alphas);

        $xT = $x->transpose();

        $coefficients = $xT->matmul($x)
            ->add($penalties)
            ->inverse()
            ->dot($xT->dot($y))
            ->asArray();

        $this->bias = (float) array_shift($coefficients);
        $this->coefficients = TEST_Vector::quick($coefficients);
    }

    /**
     * Make a prediction based on the line calculated from the training data.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return (int|float)[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->coefficients or is_null($this->bias)) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        return TEST_Matrix::build($dataset->samples())
            ->dot($this->coefficients)
            ->add($this->bias)
            ->asArray();
    }

    /**
     * Return the normalized importance scores of each feature column of the training set.
     *
     * @throws RuntimeException
     * @return float[]
     */
    public function featureImportances() : array
    {
        if (is_null($this->coefficients)) {
            throw new RuntimeException('Learner has not been trained.');
        }

        $importances = $this->coefficients->abs();

        return $importances->divide($importances->sum())->asArray();
    }
}
