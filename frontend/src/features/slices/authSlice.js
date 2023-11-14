import { createSlice } from "@reduxjs/toolkit";

const authSlice = createSlice({
    name: "auth",
    initialState: {
        data: undefined,
        email: undefined,
    },
    reducers: {
        setAuth: (state, action) => {

            const newState = { ...state }

            newState.data = action.payload

            return newState
        },
        clearAuth: (state) => {
            const newState = { ...state }

            newState.data = undefined // Set the data to undefined

            return newState
        },
        setEmail: (state, action) => {
           state.email = action.payload;
        }
    }
})

export const { setAuth, clearAuth, setEmail } = authSlice.actions
export default authSlice
