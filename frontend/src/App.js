import {Route, Routes} from 'react-router-dom';

import './App.css';
import Chat from './pages/Chat/Chat'

function App() {
    return (
        <div className="App">
            <Routes>
                <Route path='/' element={<Chat/>}/>
            </Routes>
        </div>);
}

export default App;