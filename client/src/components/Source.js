import React from "react";
// import {useNavigate,Link } from "react-router-dom"
import './Source.css'
document.body.style.backgroundImage = 'url("https://unsplash.it/777")';
function Source(){
    return(
        <>
       <h1 className="head">Choose Your Peak Hours</h1>
       <form className="form-group">
  <div className="mb-3" id="label">
    <label for="sourceInput" className="form-label">Peak Hour</label>
    <select className="mb-3" id="destination">
  <option>Holi</option>
  <option>Diwali</option>
  <option>Dussehra</option>
  <option>College/School</option>
  <option>Marriage Hours</option>
  <option>Discount Season</option>
</select>
  </div>
  <button type="submit" className="btn btn-primary">Search</button>
</form>
        </>
    )
}

export default Source;