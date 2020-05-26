<?php

namespace Test\Vendor\Rubix\ML\CrossValidation\Metrics;

use Test\Vendor\Rubix\ML\Estimator;
use Test\Vendor\Rubix\ML\EstimatorType;
use InvalidArgumentException;

use function count;

use const Test\Vendor\Rubix\ML\EPSILON;

/**
 * SMAPE
 *
 * *Symmetric Mean Absolute Percentage Error* (SMAPE) is a scale-independent regression
 * metric that expresses the relative error of a set of predictions and their labels as a
 * percentage. It is an improvement over the non-symmetric MAPE in that it is both upper
 * and lower bounded.
 *
 * References:
 * [1] V. Kreinovich. et al. How to Estimate Forecasting Quality: A System Motivated
 * Derivation of Symmetric Mean Absolute Percentage Error (SMAPE) and Other Similar
 * Characteristics.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class SMAPE implements Metric
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [-100.0, 0.0];
    }

    /**
     * The estimator types that this metric is compatible with.
     *
     * @return \Test\Vendor\Rubix\ML\EstimatorType[]
     */
    public function compatibility() : array
    {
        return [
            EstimatorType::regressor(),
        ];
    }

    /**
     * Score a set of predictions.
     *
     * @param (int|float)[] $predictions
     * @param (int|float)[] $labels
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        if (count($predictions) !== count($labels)) {
            throw new InvalidArgumentException('Number of predictions'
                . ' and labels must be equal.');
        }

        if (empty($predictions)) {
            return 0.0;
        }

        $error = 0.0;

        foreach ($predictions as $i => $prediction) {
            $label = $labels[$i];

            $error += 100.0 * abs(($prediction - $label)
                / ((abs($label) + abs($prediction)) ?: EPSILON));
        }

        return -($error / count($predictions));
    }
}
