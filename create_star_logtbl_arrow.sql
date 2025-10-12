--
-- Create star_logtbl_arrow table from Arrow IPC files using IMPORT FOREIGN SCHEMA
-- File pattern: /mnt/pgstrom-bucket/arrowdata/@{date}.arrow
--

-- Drop existing table if exists
DROP FOREIGN TABLE IF EXISTS star_logtbl_arrow;

-- Import the schema from Arrow files
-- The first parameter becomes the table name in Arrow FDW
IMPORT FOREIGN SCHEMA star_logtbl_arrow
  FROM SERVER arrow_fdw
  INTO public
  OPTIONS (dir '/mnt/pgstrom-bucket/arrowdata',
           pattern '@{date}.arrow');

-- Verify the table was created
\d star_logtbl_arrow

-- Show sample data
SELECT COUNT(*) FROM star_logtbl_arrow LIMIT 1;
