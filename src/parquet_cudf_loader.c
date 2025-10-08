/*
 * parquet_cudf_loader.c
 *
 * Dynamic loader for libpgstrom_cudf.so
 * ----
 * Copyright 2011-2025 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2025 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

/* Function pointers to cuDF library */
static void *cudf_handle = NULL;

typedef struct pgstrom_cudf_table_handle pgstrom_cudf_table_handle;

typedef struct {
	void   *null_mask;
	size_t  null_mask_size;
	void   *data;
	size_t  data_size;
	void   *offsets;
	size_t  offsets_size;
	long    num_rows;
	int     null_count;
} pgstrom_cudf_column;

/* Function pointer types */
typedef pgstrom_cudf_table_handle* (*read_parquet_func)(
	const char*, int, const int*, int, const char**);
typedef int (*get_column_func)(
	pgstrom_cudf_table_handle*, int, pgstrom_cudf_column*, const char**);
typedef long (*get_num_rows_func)(pgstrom_cudf_table_handle*);
typedef int (*get_num_columns_func)(pgstrom_cudf_table_handle*);
typedef void (*free_table_func)(pgstrom_cudf_table_handle*);

static read_parquet_func     cudf_read_parquet = NULL;
static get_column_func       cudf_get_column = NULL;
static get_num_rows_func     cudf_get_num_rows = NULL;
static get_num_columns_func  cudf_get_num_columns = NULL;
static free_table_func       cudf_free_table = NULL;

/*
 * Load libpgstrom_cudf.so dynamically
 */
int
pgstrom_cudf_load_library(const char **error_msg)
{
	static char error_buffer[512];
	const char *lib_path = "/usr/local/lib/libpgstrom_cudf.so";

	if (cudf_handle != NULL)
		return 0; /* Already loaded */

	/* Try to load the library */
	cudf_handle = dlopen(lib_path, RTLD_LAZY | RTLD_LOCAL);
	if (!cudf_handle) {
		snprintf(error_buffer, sizeof(error_buffer),
				 "Failed to load %s: %s", lib_path, dlerror());
		*error_msg = error_buffer;
		return -1;
	}

	/* Load function pointers */
	cudf_read_parquet = (read_parquet_func)dlsym(cudf_handle, "pgstrom_cudf_read_parquet");
	cudf_get_column = (get_column_func)dlsym(cudf_handle, "pgstrom_cudf_get_column");
	cudf_get_num_rows = (get_num_rows_func)dlsym(cudf_handle, "pgstrom_cudf_get_num_rows");
	cudf_get_num_columns = (get_num_columns_func)dlsym(cudf_handle, "pgstrom_cudf_get_num_columns");
	cudf_free_table = (free_table_func)dlsym(cudf_handle, "pgstrom_cudf_free_table");

	if (!cudf_read_parquet || !cudf_get_column || !cudf_get_num_rows ||
		!cudf_get_num_columns || !cudf_free_table) {
		snprintf(error_buffer, sizeof(error_buffer),
				 "Failed to resolve symbols in libpgstrom_cudf.so");
		*error_msg = error_buffer;
		dlclose(cudf_handle);
		cudf_handle = NULL;
		return -1;
	}

	fprintf(stderr, "INFO: Loaded libpgstrom_cudf.so successfully\n");
	return 0;
}

/*
 * Wrappers for cuDF functions
 */
pgstrom_cudf_table_handle *
pgstrom_cudf_read_parquet_wrapper(const char *filename,
								  int row_group_index,
								  const int *column_indices,
								  int num_columns,
								  const char **error_msg)
{
	if (cudf_handle == NULL) {
		static char msg[] = "libpgstrom_cudf.so not loaded";
		*error_msg = msg;
		return NULL;
	}
	return cudf_read_parquet(filename, row_group_index,
							 column_indices, num_columns, error_msg);
}

int
pgstrom_cudf_get_column_wrapper(pgstrom_cudf_table_handle *handle,
								int column_index,
								pgstrom_cudf_column *column_out,
								const char **error_msg)
{
	if (cudf_handle == NULL) {
		static char msg[] = "libpgstrom_cudf.so not loaded";
		*error_msg = msg;
		return -1;
	}
	return cudf_get_column(handle, column_index, column_out, error_msg);
}

long
pgstrom_cudf_get_num_rows_wrapper(pgstrom_cudf_table_handle *handle)
{
	if (cudf_handle == NULL)
		return -1;
	return cudf_get_num_rows(handle);
}

int
pgstrom_cudf_get_num_columns_wrapper(pgstrom_cudf_table_handle *handle)
{
	if (cudf_handle == NULL)
		return -1;
	return cudf_get_num_columns(handle);
}

void
pgstrom_cudf_free_table_wrapper(pgstrom_cudf_table_handle *handle)
{
	if (cudf_handle != NULL && handle != NULL)
		cudf_free_table(handle);
}
