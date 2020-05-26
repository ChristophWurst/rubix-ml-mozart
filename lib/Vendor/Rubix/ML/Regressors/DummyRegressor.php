<?php

namespace Test\Vendor\Rubix\ML\Regressors;

use Test\Vendor\Rubix\ML\Learner;
use Test\Vendor\Rubix\ML\DataType;
use Test\Vendor\Rubix\ML\Estimator;
use Test\Vendor\Rubix\ML\Persistable;
use Test\Vendor\Rubix\ML\EstimatorType;
use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Datasets\Labeled;
use Test\Vendor\Rubix\ML\Other\Strategies\Mean;
use Test\Vendor\Rubix\ML\Other\Traits\PredictsSingle;
use Test\Vendor\Rubix\ML\Other\Strategies\Continuous;
use Test\Vendor\Rubix\ML\Specifications\DatasetIsNotEmpty;
use Test\Vendor\Rubix\ML\Specifications\LabelsAreCompatibleWithLearner;
use InvalidArgumentException;
use RuntimeException;

use function count;

/**
 * Dummy Regressor
 *
 * Regressor that guesses the output values based on a Guessing Strategy. Dummy
 * Regressor is useful to provide a sanity check and to compare performance
 * against actual Regressors.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class DummyRegressor implements Estimator, Learner, Persistable
{
    use PredictsSingle;
    
    /**
     * The guessing strategy that the dummy employs.
     *
     * @var \Test\Vendor\Rubix\ML\Other\Strategies\Continuous
     */
    protected $strategy;

    /**
     * Has the learner been trained?
     *
     * @var bool
     */
    protected $trained;

    /**
     * @param \Test\Vendor\Rubix\ML\Other\Strategies\Continuous|null $strategy
     */
    public function __construct(?Continuous $strategy = null)
    {
        $this->strategy = $strategy ?? new Mean();
        $this->trained = false;
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
        return DataType::all();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'strategy' => $this->strategy,
        ];
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->trained;
    }

    /**
     * Fit the training set to the given guessing strategy.
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
        LabelsAreCompatibleWithLearner::check($dataset, $this);

        $this->strategy->fit($dataset->labels());

        $this->trained = true;
    }

    /**
     * Make a prediction of a given sample dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return (int|float)[]
     */
    public function predict(Dataset $dataset) : array
    {
        if (!$this->trained) {
            throw new RuntimeException('Estimator has not been trained.');
        }

        $n = $dataset->numRows();

        $predictions = [];

        while (count($predictions) < $n) {
            $predictions[] = $this->strategy->guess();
        }

        return $predictions;
    }
}
