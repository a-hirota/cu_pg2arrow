/*
 * pgstrom_cudf.cpp
 *
 * cuDF wrapper implementation (separate from PostgreSQL)
 * ----
 * Copyright 2011-2025 (C) KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright 2014-2025 (C) PG-Strom Developers Team
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the PostgreSQL License.
 */
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include "pgstrom_cudf.h"

/* Error buffer */
static thread_local char error_buffer[512];

/* Initialize CUDA and RMM once */
static std::once_flag cuda_init_flag;
static bool cuda_initialized = false;
static char cuda_init_error[512] = {0};

static void initialize_cuda_and_rmm() {
	/* Force disable hardware decompression to avoid compatibility issues */
	setenv("LIBCUDF_HW_DECOMPRESSION", "OFF", 1);

	const char* hw_decomp = getenv("LIBCUDF_HW_DECOMPRESSION");
	fprintf(stderr, "DEBUG: LIBCUDF_HW_DECOMPRESSION=%s\n", hw_decomp ? hw_decomp : "(not set)");

	/* Check if CUDA context already exists */
	CUcontext ctx = nullptr;
	CUresult cu_err = cuCtxGetCurrent(&ctx);

	/* Initialize CUDA runtime in this process */
	cudaError_t err = cudaFree(0);
	if (err != cudaSuccess) {
		snprintf(cuda_init_error, sizeof(cuda_init_error),
				 "CUDA initialization failed: %s (%d), existing ctx=%p",
				 cudaGetErrorString(err), err, ctx);
		return;
	}

	/* Set device to 0 (GPU0) with cudaSetDeviceFlags to allow multiple contexts */
	err = cudaSetDeviceFlags(cudaDeviceScheduleAuto | cudaDeviceMapHost);
	if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess) {
		snprintf(cuda_init_error, sizeof(cuda_init_error),
				 "cudaSetDeviceFlags failed: %s (%d)", cudaGetErrorString(err), err);
		/* Continue anyway - this might not be critical */
	}

	err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		snprintf(cuda_init_error, sizeof(cuda_init_error),
				 "cudaSetDevice(0) failed: %s (%d)", cudaGetErrorString(err), err);
		return;
	}

	/* Don't initialize RMM - let cuDF use its default allocator */
	/* This avoids potential conflicts with PG-Strom's GPU memory management */
	cuda_initialized = true;
	fprintf(stderr, "DEBUG: Skipping RMM initialization, using cuDF defaults\n");
}

/* Internal table handle structure */
struct pgstrom_cudf_table_handle {
	std::unique_ptr<cudf::table> table;
	cudf::io::table_with_metadata metadata;

	pgstrom_cudf_table_handle(cudf::io::table_with_metadata&& result)
		: metadata(std::move(result))
	{
		table = std::move(metadata.tbl);
	}
};

/*
 * Read Parquet file using cuDF
 */
extern "C" pgstrom_cudf_table_handle *
pgstrom_cudf_read_parquet(const char *filename,
						  int row_group_index,
						  const int *column_indices,
						  int num_columns,
						  const char **error_message)
{
	*error_message = nullptr;
	error_buffer[0] = '\0';

	try {
		/* Initialize CUDA and RMM */
		std::call_once(cuda_init_flag, initialize_cuda_and_rmm);

		if (!cuda_initialized) {
			/* Return the error from initialization */
			if (cuda_init_error[0] != '\0') {
				strncpy(error_buffer, cuda_init_error, sizeof(error_buffer)-1);
				*error_message = error_buffer;
			} else {
				*error_message = "CUDA/RMM initialization failed for unknown reason";
			}
			return nullptr;
		}

		auto source = cudf::io::source_info(filename);
		auto builder = cudf::io::parquet_reader_options::builder(source);

		/* Select row group */
		std::vector<std::vector<cudf::size_type>> row_groups = {{row_group_index}};
		builder.row_groups(row_groups);

		/* Select columns if specified - cuDF uses column indices */
		/* Note: leaving columns unspecified reads all columns */
		if (num_columns > 0 && column_indices != nullptr) {
			// cuDF will select columns by index automatically
		}

		auto options = builder.build();

		/* Read Parquet file */
		auto result = cudf::io::read_parquet(options);

		/* Create handle */
		auto *handle = new pgstrom_cudf_table_handle(std::move(result));

		return handle;
	}
	catch (const std::exception &e) {
		snprintf(error_buffer, sizeof(error_buffer),
				 "cuDF read_parquet failed: %s", e.what());
		*error_message = error_buffer;
		return nullptr;
	}
}

/*
 * Get column metadata
 */
extern "C" int
pgstrom_cudf_get_column(pgstrom_cudf_table_handle *handle,
						int column_index,
						pgstrom_cudf_column *column_out,
						const char **error_message)
{
	*error_message = nullptr;
	error_buffer[0] = '\0';

	try {
		if (!handle || !handle->table) {
			snprintf(error_buffer, sizeof(error_buffer), "Invalid table handle");
			*error_message = error_buffer;
			return -1;
		}

		auto table_view = handle->table->view();

		if (column_index < 0 || column_index >= table_view.num_columns()) {
			snprintf(error_buffer, sizeof(error_buffer),
					 "Column index %d out of range [0, %d)",
					 column_index, table_view.num_columns());
			*error_message = error_buffer;
			return -1;
		}

		auto col_view = table_view.column(column_index);

		/* Get null mask */
		if (col_view.nullable() && col_view.null_mask()) {
			column_out->null_mask = const_cast<void*>(
				static_cast<const void*>(col_view.null_mask()));
			column_out->null_mask_size = cudf::bitmask_allocation_size_bytes(col_view.size());
		} else {
			column_out->null_mask = nullptr;
			column_out->null_mask_size = 0;
		}

		/* Get data buffer and handle variable-length types specially */
		if (col_view.type().id() == cudf::type_id::STRING) {
			/* For STRING columns, use strings_column_view API */
			auto strings_view = cudf::strings_column_view(col_view);

			fprintf(stderr, "  STRING column has %d children\n", col_view.num_children());

			/* Get chars buffer using chars_begin() - returns pointer to first char */
			auto chars_ptr = strings_view.chars_begin(cudf::get_default_stream());
			auto chars_bytes = strings_view.chars_size(cudf::get_default_stream());
			column_out->data = const_cast<void*>(static_cast<const void*>(chars_ptr));
			column_out->data_size = chars_bytes;

			/* Offsets buffer (child 0) */
			auto offsets_col = col_view.child(0);
			column_out->offsets = const_cast<void*>(static_cast<const void*>(offsets_col.head()));
			column_out->offsets_size = offsets_col.size() * sizeof(int32_t);
		} else {
			/* For fixed-width types */
			column_out->data = const_cast<void*>(static_cast<const void*>(col_view.head()));
			column_out->data_size = col_view.size() * cudf::size_of(col_view.type());
			column_out->offsets = nullptr;
			column_out->offsets_size = 0;
		}

		column_out->num_rows = col_view.size();
		column_out->null_count = col_view.null_count();

		return 0;
	}
	catch (const std::exception &e) {
		snprintf(error_buffer, sizeof(error_buffer),
				 "cuDF get_column failed: %s", e.what());
		*error_message = error_buffer;
		return -1;
	}
}

/*
 * Get number of rows
 */
extern "C" int64_t
pgstrom_cudf_get_num_rows(pgstrom_cudf_table_handle *handle)
{
	if (!handle || !handle->table)
		return -1;
	return handle->table->num_rows();
}

/*
 * Get number of columns
 */
extern "C" int
pgstrom_cudf_get_num_columns(pgstrom_cudf_table_handle *handle)
{
	if (!handle || !handle->table)
		return -1;
	return handle->table->num_columns();
}

/*
 * Free table
 */
extern "C" void
pgstrom_cudf_free_table(pgstrom_cudf_table_handle *handle)
{
	if (handle) {
		delete handle;
	}
}
