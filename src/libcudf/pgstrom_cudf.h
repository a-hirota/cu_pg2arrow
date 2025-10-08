/*
 * pgstrom_cudf.h
 *
 * C API for PG-Strom cuDF integration (separate library)
 * ----
 * Copyright 2011-2025 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2025 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#ifndef PGSTROM_CUDF_H
#define PGSTROM_CUDF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * cuDF Column Buffer - describes a single column's GPU buffers
 */
typedef struct {
	void   *null_mask;       /* GPU pointer to null bitmap */
	size_t  null_mask_size;
	void   *data;            /* GPU pointer to data buffer */
	size_t  data_size;
	void   *offsets;         /* GPU pointer to offsets (for strings/lists) */
	size_t  offsets_size;
	int64_t num_rows;
	int     null_count;
} pgstrom_cudf_column;

/*
 * cuDF Table Handle - opaque handle to cuDF table on GPU
 */
typedef struct pgstrom_cudf_table_handle pgstrom_cudf_table_handle;

/*
 * Read Parquet file row group using cuDF, returns table handle
 * Keeps data on GPU.
 */
pgstrom_cudf_table_handle *pgstrom_cudf_read_parquet(
	const char *filename,
	int row_group_index,
	const int *column_indices,
	int num_columns,
	const char **error_message);

/*
 * Get column metadata from cuDF table
 */
int pgstrom_cudf_get_column(
	pgstrom_cudf_table_handle *handle,
	int column_index,
	pgstrom_cudf_column *column_out,
	const char **error_message);

/*
 * Get number of rows in table
 */
int64_t pgstrom_cudf_get_num_rows(
	pgstrom_cudf_table_handle *handle);

/*
 * Get number of columns in table
 */
int pgstrom_cudf_get_num_columns(
	pgstrom_cudf_table_handle *handle);

/*
 * Free cuDF table and associated GPU memory
 */
void pgstrom_cudf_free_table(
	pgstrom_cudf_table_handle *handle);

#ifdef __cplusplus
}
#endif

#endif /* PGSTROM_CUDF_H */
