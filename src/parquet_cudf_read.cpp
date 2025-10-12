/*
 * parquet_cudf_read.cpp
 *
 * GPU-native Parquet reading using cuDF (via dynamic loader)
 *
 * This module uses NVIDIA RAPIDS cuDF library to decompress Parquet files
 * directly on GPU and converts the result to PG-Strom's KDS format.
 *
 * ----
 * Copyright 2011-2025 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2025 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <vector>

#include <cuda_runtime.h>
#include "xpu_common.h"
#include "arrow_defs.h"
#include "parquet_cudf_loader.h"

/* Error reporting */
static __thread char __error_buffer[512];
#define SET_ERROR(fmt, ...) \
	snprintf(__error_buffer, sizeof(__error_buffer), \
			 "[%s:%d] " fmt, __FILE__, __LINE__, ##__VA_ARGS__)

/*
 * Read Parquet row group using cuDF and convert to KDS
 *
 * This is a regular C++ function (not extern "C") since it's called from
 * parquet_read.cpp within the same module.
 *
 * The __attribute__((used)) prevents LTO from removing this function even
 * with -fvisibility=hidden.
 */
__attribute__((used))
kern_data_store *
parquetReadRowGroupCuDF(const char *filename,
						int row_group_index,
						const std::vector<int> &column_indices,
						const kern_data_store *kds_head,
						void *(*malloc_callback)(void *malloc_private, size_t malloc_size),
						void *malloc_private,
						const char **p_error_message)
{
	*__error_buffer = '\0';
	*p_error_message = __error_buffer;

	/* Use the vector directly */
	int num_cols = column_indices.size();

	/* Load cuDF library if not already loaded */
	const char *load_error = NULL;
	if (pgstrom_cudf_load_library(&load_error) != 0) {
		SET_ERROR("Failed to load cuDF library: %s", load_error ? load_error : "unknown");
		return NULL;
	}

	try {
		/* Read Parquet with cuDF */
		const char *cudf_error = NULL;
		pgstrom_cudf_table_handle *table = pgstrom_cudf_read_parquet_wrapper(
			filename,
			row_group_index,
			column_indices.data(),
			num_cols,
			&cudf_error);

		if (!table) {
			SET_ERROR("cuDF read_parquet failed: %s", cudf_error ? cudf_error : "unknown");
			return NULL;
		}

		int64_t num_rows = pgstrom_cudf_get_num_rows_wrapper(table);
		int num_columns = pgstrom_cudf_get_num_columns_wrapper(table);

		/* Estimate KDS buffer size */
		size_t kds_length = KDS_HEAD_LENGTH(kds_head);

		for (int k = 0; k < num_columns; k++) {
			pgstrom_cudf_column col;
			if (pgstrom_cudf_get_column_wrapper(table, k, &col, &cudf_error) != 0) {
				pgstrom_cudf_free_table_wrapper(table);
				SET_ERROR("Failed to get column %d: %s", k, cudf_error ? cudf_error : "unknown");
				return NULL;
			}

			kds_length += ARROW_ALIGN(col.null_mask_size);
			kds_length += ARROW_ALIGN(col.data_size);
			kds_length += ARROW_ALIGN(col.offsets_size);
		}

		/* Allocate KDS buffer */
		kern_data_store *kds = (kern_data_store *)malloc_callback(malloc_private, kds_length);
		if (!kds) {
			pgstrom_cudf_free_table_wrapper(table);
			SET_ERROR("Out of memory for KDS buffer");
			return NULL;
		}

		/* Initialize KDS header and virtual column data */
		size_t head_copy_size = KDS_HEAD_LENGTH(kds_head) + kds_head->arrow_virtual_usage;
		memcpy(kds, kds_head, head_copy_size);
		kds->length = kds_length;
		kds->nitems = num_rows;
		kds->format = KDS_FORMAT_ARROW;

		size_t curr_pos = head_copy_size;

		/* Copy column data from GPU to host KDS */
		for (int k = 0; k < num_columns; k++) {
			pgstrom_cudf_column col;
			if (pgstrom_cudf_get_column_wrapper(table, k, &col, &cudf_error) != 0) {
				pgstrom_cudf_free_table_wrapper(table);
				SET_ERROR("Failed to get column %d: %s", k, cudf_error ? cudf_error : "unknown");
				return NULL;
			}

			int field_idx = column_indices[k];

			/* Find KDS column index for this field_index */
			int col_idx = -1;
			for (int j = 0; j < kds_head->ncols; j++) {
				if (kds_head->colmeta[j].field_index == field_idx) {
					col_idx = j;
					break;
				}
			}
			if (col_idx < 0) {
				pgstrom_cudf_free_table_wrapper(table);
				SET_ERROR("Field index %d not found in KDS (column %d)", field_idx, k);
				return NULL;
			}

			auto cmeta = &kds->colmeta[col_idx];

			/* Copy null bitmap from GPU */
			if (col.null_mask && col.null_mask_size > 0) {
				cmeta->nullmap_offset = curr_pos;
				cmeta->nullmap_length = col.null_mask_size;

				cudaError_t err = cudaMemcpy((char *)kds + curr_pos,
											 col.null_mask,
											 col.null_mask_size,
											 cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) {
					pgstrom_cudf_free_table_wrapper(table);
					SET_ERROR("cudaMemcpy failed for null bitmap: %s", cudaGetErrorString(err));
					return NULL;
				}
				curr_pos += ARROW_ALIGN(col.null_mask_size);
			} else {
				cmeta->nullmap_offset = 0;
				cmeta->nullmap_length = 0;
			}

			/* Handle STRING/variable-length vs fixed-width columns differently */
			if (col.offsets && col.offsets_size > 0) {
				/* Variable-length column (STRING, etc.) */
				cmeta->values_offset = curr_pos;
				cmeta->values_length = col.offsets_size;

				cudaError_t err = cudaMemcpy((char *)kds + curr_pos,
											 col.offsets,
											 col.offsets_size,
											 cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) {
					pgstrom_cudf_free_table_wrapper(table);
					SET_ERROR("cudaMemcpy failed for offset buffer: %s", cudaGetErrorString(err));
					return NULL;
				}
				curr_pos += ARROW_ALIGN(col.offsets_size);

				/* Copy chars buffer to extra_offset */
				if (col.data && col.data_size > 0) {
					cmeta->extra_offset = curr_pos;
					cmeta->extra_length = col.data_size;

					err = cudaMemcpy((char *)kds + curr_pos,
									 col.data,
									 col.data_size,
									 cudaMemcpyDeviceToHost);
					if (err != cudaSuccess) {
						pgstrom_cudf_free_table_wrapper(table);
						SET_ERROR("cudaMemcpy failed for chars buffer: %s", cudaGetErrorString(err));
						return NULL;
					}
					curr_pos += ARROW_ALIGN(col.data_size);
				} else {
					cmeta->extra_offset = 0;
					cmeta->extra_length = 0;
				}
			} else {
				/* Fixed-width column (INT, DOUBLE, etc.) */
				if (col.data && col.data_size > 0) {
					cmeta->values_offset = curr_pos;
					cmeta->values_length = col.data_size;

					cudaError_t err = cudaMemcpy((char *)kds + curr_pos,
												 col.data,
												 col.data_size,
												 cudaMemcpyDeviceToHost);
					if (err != cudaSuccess) {
						pgstrom_cudf_free_table_wrapper(table);
						SET_ERROR("cudaMemcpy failed for data buffer: %s", cudaGetErrorString(err));
						return NULL;
					}
					curr_pos += ARROW_ALIGN(col.data_size);
				} else {
					cmeta->values_offset = 0;
					cmeta->values_length = 0;
				}
				cmeta->extra_offset = 0;
				cmeta->extra_length = 0;
			}
		}

		kds->usage = curr_pos;

		/* Cleanup cuDF table */
		pgstrom_cudf_free_table_wrapper(table);

		return kds;
	}
	catch (const std::exception &e) {
		SET_ERROR("Exception: %s", e.what());
		return NULL;
	}
}
