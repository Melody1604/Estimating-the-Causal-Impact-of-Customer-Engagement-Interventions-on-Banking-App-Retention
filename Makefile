.PHONY: install data dbt pipeline test all clean

install:
	python -m pip install -r requirements.txt

data:
	python -m src.data.generate_synthetic

dbt:
	dbt seed --project-dir dbt --profiles-dir dbt
	dbt run --project-dir dbt --profiles-dir dbt
	dbt test --project-dir dbt --profiles-dir dbt

pipeline:
	python -m src.pipeline

test:
	pytest -q

all: data pipeline test

clean:
	rm -rf data/raw/*.csv data/processed/* reports/tables/* reports/figures/* models/* dbt/target dbt/logs dbt/banking.duckdb
