<script>
  import { VegaLite } from 'svelte-vega';

  // Data from genres_df_f
  let data = [
    { Genre: 'Drama', Count_f: 2332 },
    { Genre: 'Black-and-white', Count_f: 298 },
    { Genre: 'Adventure', Count_f: 364 },
    { Genre: 'World Cinema', Count_f: 1269 },
    { Genre: 'Independent', Count_f: 98 },
    { Genre: 'Comedy', Count_f: 1198 },
    { Genre: 'War and Military', Count_f: 105 },
    { Genre: 'Biographical and Real-Life Inspired', Count_f: 197 },
    { Genre: 'Romance', Count_f: 875 },
    { Genre: 'Crime and Mystery', Count_f: 345 },
    { Genre: 'Action', Count_f: 514 },
    { Genre: 'Historical Films', Count_f: 70 },
    { Genre: 'Documentary', Count_f: 181 },
    { Genre: 'Music and Dance', Count_f: 317 },
    { Genre: 'Science Fiction', Count_f: 94 },
    { Genre: 'Horror', Count_f: 397 },
    { Genre: 'Thriller', Count_f: 467 },
    { Genre: 'Western', Count_f: 83 },
    { Genre: 'Artistic', Count_f: 59 },
    { Genre: 'Feminist', Count_f: 35 },
    { Genre: 'Television movie', Count_f: 65 },
    { Genre: 'LGBT', Count_f: 128 },
    { Genre: "Family's Films", Count_f: 250 },
    { Genre: 'Short Film', Count_f: 142 },
    { Genre: 'Animation', Count_f: 92 },
    { Genre: 'Erotic and Adult Films', Count_f: 49 },
    { Genre: 'Religious Film', Count_f: 9 },
    { Genre: 'Silent film', Count_f: 72 },
    { Genre: 'Reboot', Count_f: 117 },
    { Genre: 'Fantasy', Count_f: 109 },
    { Genre: "Children's Films", Count_f: 56 },
    { Genre: 'Slice of life story', Count_f: 15 },
    { Genre: 'Camp', Count_f: 2 },
    { Genre: 'Film', Count_f: 15 },
    { Genre: 'Holiday Movie', Count_f: 10 },
    { Genre: 'Theatrical Movie', Count_f: 6 },
    { Genre: 'Christian film', Count_f: 8 },
    { Genre: 'Pre-Code', Count_f: 11 },
    { Genre: 'Other Genres', Count_f: 10 },
    { Genre: 'Business', Count_f: 1 },
    { Genre: 'Finance & Investing', Count_f: 1 },
    { Genre: 'Nature', Count_f: 1 },
    { Genre: 'Female buddy film', Count_f: 1 },
    { Genre: 'Blaxploitation', Count_f: 1 },
    { Genre: 'Movie serial', Count_f: 1 }
  ];

  // Calculate proportions
  const total = data.reduce((sum, d) => sum + d.Count_f, 0);
  data = data.map(d => ({
    ...d,
    Proportion: d.Count_f / total
  }));

  // Vega-Lite specification for a layer chart
  const spec = {
    width: 400,
    height: 400,
    data: { values: data },
    layer: [
      {
        mark: { type: 'arc', innerRadius: 50 },
        encoding: {
          theta: { field: 'Count_f', type: 'quantitative' },
          color: { field: 'Genre', type: 'nominal' },
          tooltip: [
            { field: 'Genre', type: 'nominal', title: 'Genre' },
            { field: 'Count_f', type: 'quantitative', title: 'Count' },
            {
              field: 'Proportion',
              type: 'quantitative',
              title: 'Percentage',
              format: '.1%' // Format as percentage
            }
          ]
        }
      },
      {
        mark: { type: 'text', radius: 150 },
        encoding: {
          text: { field: 'Genre', type: 'nominal' },
          theta: { field: 'Count_f', type: 'quantitative' },
          color: { value: 'black' } // Adjust text color as needed
        }
      }
    ]
  };
</script>

<style>
  /* Optional styling */
  .chart-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
  }
</style>

<div class="chart-container">
  <VegaLite spec={spec} />
</div>
