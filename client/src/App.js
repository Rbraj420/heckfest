import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
// import Front from './components/Front';
import Source from './components/Source';
import Api from './components/Api';
// import './App.css';
function App() {
  return (
<>
<BrowserRouter>
<Routes>
  {/* <Route exact path="/" element={<Front/>} /> */}
  <Route exact path="/" element={<Api/>} />
  <Route exact path="/source" element={<Source/>} />
</Routes>
</BrowserRouter>
</>
  );
}

export default App;
