/*
 * Minimal cuDF wrapper - no CUDA/RMM initialization
 */
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include "pgstrom_cudf.h"

static thread_local char error_buffer[512];

struct pgstrom_cudf_table_handle {
	std::unique_ptr<cudf::table> table;
	cudf::io::table_with_metadata metadata;
	
	pgstrom_cudf_table_handle(cudf::io::table_with_metadata&& result)
		: metadata(std::move(result))
	{
		table = std::move(metadata.tbl);
	}
};

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
		// Simple: just read the file
		auto source = cudf::io::source_info(filename);
		std::vector<std::vector<cudf::size_type>> row_groups = {{row_group_index}};
		auto options = cudf::io::parquet_reader_options::builder(source)
			.row_groups(row_groups)
			.build();
		
		auto result = cudf::io::read_parquet(options);
		auto *handle = new pgstrom_cudf_table_handle(std::move(result));
		
		return handle;
	}
	catch (const std::exception &e) {
		snprintf(error_buffer, sizeof(error_buffer), "cuDF error: %s", e.what());
		*error_message = error_buffer;
		return nullptr;
	}
}

extern "C" int64_t
pgstrom_cudf_get_num_rows(pgstrom_cudf_table_handle *handle)
{
	return handle->table->num_rows();
}

extern "C" int
pgstrom_cudf_get_num_columns(pgstrom_cudf_table_handle *handle)
{
	return handle->table->num_columns();
}

extern "C" int
pgstrom_cudf_get_column(pgstrom_cudf_table_handle *handle,
						int column_index,
						pgstrom_cudf_column *column_out,
						const char **error_message)
{
	try {
		auto &col_view = handle->table->get_column(column_index).view();
		
		column_out->null_mask = const_cast<void*>(static_cast<const void*>(col_view.null_mask()));
		column_out->null_mask_size = cudf::bitmask_allocation_size_bytes(col_view.size());
		column_out->num_rows = col_view.size();
		column_out->null_count = col_view.null_count();
		
		if (col_view.type().id() == cudf::type_id::STRING) {
			auto strings_view = cudf::strings_column_view(col_view);
			auto chars_ptr = strings_view.chars_begin(cudf::get_default_stream());
			column_out->data = const_cast<void*>(static_cast<const void*>(chars_ptr));
			column_out->data_size = strings_view.chars_size(cudf::get_default_stream());
			
			auto offsets_col = col_view.child(0);
			column_out->offsets = const_cast<void*>(static_cast<const void*>(offsets_col.head()));
			column_out->offsets_size = offsets_col.size() * sizeof(int32_t);
		} else {
			column_out->data = const_cast<void*>(static_cast<const void*>(col_view.head()));
			column_out->data_size = col_view.size() * cudf::size_of(col_view.type());
			column_out->offsets = nullptr;
			column_out->offsets_size = 0;
		}
		
		return 0;
	}
	catch (const std::exception &e) {
		snprintf(error_buffer, sizeof(error_buffer), "cuDF get_column error: %s", e.what());
		*error_message = error_buffer;
		return -1;
	}
}

extern "C" void
pgstrom_cudf_free_table(pgstrom_cudf_table_handle *handle)
{
	delete handle;
}
