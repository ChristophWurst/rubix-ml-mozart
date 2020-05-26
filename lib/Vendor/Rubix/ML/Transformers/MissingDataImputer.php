<?php

namespace Test\Vendor\Rubix\ML\Transformers;

use Test\Vendor\Rubix\ML\DataType;
use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Other\Strategies\Mean;
use Test\Vendor\Rubix\ML\Other\Strategies\Continuous;
use Test\Vendor\Rubix\ML\Other\Strategies\Categorical;
use Test\Vendor\Rubix\ML\Other\Strategies\KMostFrequent;
use Test\Vendor\Rubix\ML\Specifications\SamplesAreCompatibleWithTransformer;
use InvalidArgumentException;
use RuntimeException;

use function is_null;

/**
 * Missing Data Imputer
 *
 * Missing Data Imputer replaces missing continuous (denoted by `NaN`) or categorical values
 * (denoted by special placeholder category) with a guess based on user-defined Strategy.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MissingDataImputer implements Transformer, Stateful
{
    /**
     * The guessing strategy to use when imputing continuous values.
     *
     * @var \Test\Vendor\Rubix\ML\Other\Strategies\Continuous
     */
    protected $continuous;

    /**
     * The guessing strategy to use when imputing categorical values.
     *
     * @var \Test\Vendor\Rubix\ML\Other\Strategies\Categorical
     */
    protected $categorical;

    /**
     * The placeholder category that denotes missing values.
     *
     * @var string
     */
    protected $categoricalPlaceholder;

    /**
     * The fitted guessing strategy for each feature column.
     *
     * @var mixed[]|null
     */
    protected $strategies;

    /**
     * The data types of the fitted feature columns.
     *
     * @var \Test\Vendor\Rubix\ML\DataType[]|null
     */
    protected $types;

    /**
     * @param \Test\Vendor\Rubix\ML\Other\Strategies\Continuous|null $continuous
     * @param \Test\Vendor\Rubix\ML\Other\Strategies\Categorical|null $categorical
     * @param string $categoricalPlaceholder
     * @throws \InvalidArgumentException
     */
    public function __construct(
        ?Continuous $continuous = null,
        ?Categorical $categorical = null,
        string $categoricalPlaceholder = '?'
    ) {
        $this->continuous = $continuous ?? new Mean();
        $this->categorical = $categorical ?? new KMostFrequent(1);
        $this->categoricalPlaceholder = $categoricalPlaceholder;
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @return \Test\Vendor\Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return DataType::all();
    }

    /**
     * Is the transformer fitted?
     *
     * @return bool
     */
    public function fitted() : bool
    {
        return isset($this->strategies) and isset($this->types);
    }

    /**
     * Fit the transformer to a dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \InvalidArgumentException
     */
    public function fit(Dataset $dataset) : void
    {
        SamplesAreCompatibleWithTransformer::check($dataset, $this);
        
        $this->strategies = $this->types = [];

        foreach ($dataset->columnTypes() as $column => $type) {
            $donors = [];

            switch ($type->code()) {
                case DataType::CONTINUOUS:
                    $strategy = clone $this->continuous;

                    foreach ($dataset->column($column) as $value) {
                        if (is_float($value) and is_nan($value)) {
                            continue 1;
                        }
        
                        $donors[] = $value;
                    }

                    break 1;

                case DataType::CATEGORICAL:
                    $strategy = clone $this->categorical;

                    foreach ($dataset->column($column) as $value) {
                        if ($value !== $this->categoricalPlaceholder) {
                            $donors[] = $value;
                        }
                    }

                    break 1;
            }

            if (!isset($strategy)) {
                continue 1;
            }

            if (empty($donors)) {
                throw new InvalidArgumentException('Dataset must contain'
                    . ' at least 1 donor per feature column.');
            }

            $strategy->fit($donors);

            $this->strategies[$column] = $strategy;
            $this->types[$column] = $type;
        }
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     * @throws \RuntimeException
     */
    public function transform(array &$samples) : void
    {
        if (is_null($this->strategies) or is_null($this->types)) {
            throw new RuntimeException('Transformer has not been fitted.');
        }

        foreach ($samples as &$sample) {
            foreach ($this->types as $column => $type) {
                $value = &$sample[$column];

                switch ($type->code()) {
                    case DataType::CONTINUOUS:
                        if (is_float($value) and is_nan($value)) {
                            $value = $this->strategies[$column]->guess();
                        }

                        break 1;

                    case DataType::CATEGORICAL:
                        if ($value === $this->categoricalPlaceholder) {
                            $value = $this->strategies[$column]->guess();
                        }

                        break 1;
                }
            }
        }
    }
}
