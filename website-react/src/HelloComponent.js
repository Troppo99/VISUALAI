import React from 'react';

function HelloComponent(props) {
  return (
    // p-4 bg-blue-100 text-blue-700 font-bold rounded
    <div className="text-blue-500 p-4 bg-blue-100 font-bold rounded">
      Hello, {props.name}!
    </div>
  );
}

export default HelloComponent;
