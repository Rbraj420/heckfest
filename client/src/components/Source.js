import React from "react";
// import {useNavigate,Link } from "react-router-dom"
import './Source.css'
document.body.style.backgroundImage = 'url("https://unsplash.it/777")';
function Source(){
    return(
        <>
       <h1 className="head">Choose Your Routes</h1>
       <form className="form-group">
  <div className="mb-3" id="label">
    <label for="sourceInput" className="form-label">Source</label>
    <select className="mb-3" id="destination">
  <option>Dhanbad</option>
  <option>Patna</option>
  <option>Delhi</option>
  <option>Ayodhya Dham</option>
  <option>Varanasi</option>
</select>
  </div>
  <div className="mb-3"id="label">
    <label for="destinationInput" className="form-label">Destination</label>
    <select className="mb-3" id="destination">
  <option>Dhanbad</option>
  <option>Patna</option>
  <option>Delhi</option>
  <option>Ayodhya Dham</option>
  <option>Varanasi</option>
</select>
  </div>
  <div className="mb-3"id="label" >
  <label for="date" className="form-label">Past Record</label>
  <select className="mb-3" id="date">
  <option>Week</option>
  <option>Month</option>
  <option>Quarter Year</option>
  <option>Half Year</option>
  <option>Year</option>
</select>
</div>
  <button type="submit" className="btn btn-primary">Search</button>
</form>
        </>
    )
}

export default Source;