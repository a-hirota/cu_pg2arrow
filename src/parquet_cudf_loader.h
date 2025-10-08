/*
 * parquet_cudf_loader.h
 *
 * Header for cuDF dynamic loader
 */
#ifndef PARQUET_CUDF_LOADER_H
#define PARQUET_CUDF_LOADER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct pgstrom_cudf_table_handle pgstrom_cudf_table_handle;

/* Column metadata */
typedef struct {
	void   *null_mask;
	size_t  null_mask_size;
	void   *data;
	size_t  data_size;
	void   *offsets;
	size_t  offsets_size;
	int64_t num_rows;
	int     null_count;
} pgstrom_cudf_column;

/* Load library */
int pgstrom_cudf_load_library(const char **error_msg);

/* Wrapper functions */
pgstrom_cudf_table_handle *pgstrom_cudf_read_parquet_wrapper(
	const char *filename,
	int row_group_index,
	const int *column_indices,
	int num_columns,
	const char **error_msg);

int pgstrom_cudf_get_column_wrapper(
	pgstrom_cudf_table_handle *handle,
	int column_index,
	pgstrom_cudf_column *column_out,
	const char **error_msg);

int64_t pgstrom_cudf_get_num_rows_wrapper(
	pgstrom_cudf_table_handle *handle);

int pgstrom_cudf_get_num_columns_wrapper(
	pgstrom_cudf_table_handle *handle);

void pgstrom_cudf_free_table_wrapper(
	pgstrom_cudf_table_handle *handle);

#ifdef __cplusplus
}
#endif

#endif /* PARQUET_CUDF_LOADER_H */
