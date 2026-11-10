# Data Dictionary

## Dataset overview

The supplied files describe mobile-device specifications.

| Asset | Shape | Purpose |
|---|---:|---|
| \`data/raw/train.csv\` | 2,000 × 21 | Labeled training data |
| \`data/raw/test.csv\` | 1,000 × 21 | Unlabeled inference data |

The training data has 20 predictors and one target (\`price_range\`). The test data has an \`id\` identifier and the same 20 predictors. The model excludes \`id\` from training and inference features.

## Target

| Column | Type | Description |
|---|---|---|
| \`price_range\` | integer | Encoded class label with observed values 0, 1, 2, and 3 |

The supplied files do not define the business meaning or currency thresholds for each encoded class. Do not interpret the integers as specific prices.

## Predictors

Descriptions below follow conventional interpretations of the supplied column names. Exact units and collection definitions were not provided with the files.

| Column | Type | Working description |
|---|---|---|
| \`battery_power\` | integer | Battery capacity/power measure |
| \`blue\` | integer/binary | Bluetooth availability indicator |
| \`clock_speed\` | float | Processor clock-speed measure |
| \`dual_sim\` | integer/binary | Dual-SIM capability indicator |
| \`fc\` | integer | Front-camera specification |
| \`four_g\` | integer/binary | 4G capability indicator |
| \`int_memory\` | integer | Internal memory measure |
| \`m_dep\` | float | Mobile depth measure |
| \`mobile_wt\` | integer | Device weight measure |
| \`n_cores\` | integer | Processor core count |
| \`pc\` | integer | Primary/rear-camera specification |
| \`px_height\` | integer | Pixel height |
| \`px_width\` | integer | Pixel width |
| \`ram\` | integer | RAM measure |
| \`sc_h\` | integer | Screen height measure |
| \`sc_w\` | integer | Screen width measure |
| \`talk_time\` | integer | Talk-time measure |
| \`three_g\` | integer/binary | 3G capability indicator |
| \`touch_screen\` | integer/binary | Touchscreen indicator |
| \`wifi\` | integer/binary | Wi-Fi availability indicator |

## Data quality checks performed

| Check | Training | Test |
|---|---:|---:|
| Missing cells | 0 | 0 |
| Exact duplicate rows | 0 | 0 |
| Row count | 2,000 | 1,000 |
| Column count | 21 | 21 |

Training target distribution:

| Encoded class | Rows | Share |
|---:|---:|---:|
| 0 | 500 | 25% |
| 1 | 500 | 25% |
| 2 | 500 | 25% |
| 3 | 500 | 25% |

## Provenance and caveats

The CSV files were provided directly for this project. No source URL, license, data-collection notes, formal schema, or feature-unit specification accompanied them. Descriptions in this document are therefore working interpretations, not independently verified metadata.
