import React, { useState } from 'react';

function Api() {
  const [invoiceDate, setInvoiceDate] = useState('');
  const [prediction, setPrediction] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch('/predict-units-sold', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ invoice_date: invoiceDate })
      });

      if (!response.ok) {
        throw new Error('Failed to get prediction');
      }

      const data = await response.json();
      setPrediction(data.predicted_units_sold);
      setError('');
    } catch (error) {
      console.error('Error:', error);
      setError('Failed to get prediction');
      setPrediction('');
    }
  };

  return (
    <div>
      <h1>Predict Units Sold</h1>
      <form onSubmit={handleSubmit}>
        <label htmlFor="invoiceDate">Invoice Date:</label>
        <input
          type="text"
          id="invoiceDate"
          value={invoiceDate}
          onChange={(e) => setInvoiceDate(e.target.value)}
        />
        <button type="submit">Predict</button>
      </form>
      {error && <p>Error: {error}</p>}
      {prediction && <p>Predicted Units Sold: {prediction}</p>}
    </div>
  );
}

export default Api;
