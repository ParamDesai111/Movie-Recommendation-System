-- Make table for cleaned data with the cleaned plot, title, and poster
-- CREATE TABLE cleaned_data (
--     title NVARCHAR(255) NOT NULL,
--     cleaned_plot NVARCHAR(MAX) NOT NULL,
--     poster NVARCHAR(MAX) NOT NULL
-- );

SELECT * FROM cleaned_data
WHERE title LIKE '%';

-- TRUNCATE TABLE cleaned_data;