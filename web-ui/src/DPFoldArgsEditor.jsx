
import React, {useReducer} from 'react'
import {useApi} from "@web-gasket/ApiSession.jsx";
import {DefaultButton} from "@web-gasket/widgets/Buttons.jsx"

const canSaveFunc = state => {

    return state.cc_cluster && state.cc_allocation && state.cc_project
}

const dpFoldArgsReducer = (state, action) => {
    switch (action.name) {
        case "init": {
            return {
                args: {
                    cc_cluster: null,
                    cc_allocation: null,
                    cc_project: null
                },
                isDirty: false,
                canSave: false
            }
        }
        case "update": {

            const nextState = {
                ...state,
                args: {
                    ...state.args,
                    [action.fieldName]: action.value
                }
            }

            const canSave = canSaveFunc(nextState.args)

            return {
                ...nextState,
                canSave,
                isDirty: true
            }
        }
        case "saved": {
            return {
                ...state,
                isDirty: false
            }
        }
    }
}



const ClusterSelect = ({value, onSelect}) => {

    return <>
        <label
            className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
        >Select a cluster</label>
        <select
            className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
            onChange={e => onSelect(e.target.value)}
        >
            <option value={""}>Choose a cluster</option>
            <option value="narval">Narval</option>
            <option value="beluga">Beluga</option>
            <option value="cedar">Cedar</option>
            <option value="graham">Graham</option>
        </select>
        </>
}


const DPFoldArgsEditor = ({pipelineInstance, pipelineArgsDispatcher}) => {

    const [dpFoldArgsView, dpFoldArgsDispatcher] = useReducer(
        dpFoldArgsReducer,
        dpFoldArgsReducer(null, {
            name: 'init',
            schema: pipelineInstance.argsSchema,
            args: pipelineInstance.args
        })
    )

    const api = useApi()

    const saveArgs = () => {
        api.savePipelineInstanceArgs(pipelineInstance.pid, dpFoldArgsView.args).then(() =>
            pipelineArgsDispatcher({name: "saved"})
        )
    }

    return <div className="grid gap-6 mb-6 md:grid-cols-1">
            <div>
                <ClusterSelect
                    value={dpFoldArgsView.args.cluster}
                    onSelect={cluster => dpFoldArgsDispatcher({
                        name: "update",
                        fieldName: "cc_cluster",
                        value: cluster
                    })}
                />
            </div>
            <div>
                <label className="block mb-2 text-sm font-medium text-gray-900 dark:text-white">
                    compute allocation
                </label>
                <input
                    type="text"
                    className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                    placeholder="def-my-allocation"
                    value={dpFoldArgsView.args.cc_allocation || ""}
                    onChange={e => dpFoldArgsDispatcher({
                        name: "update",
                        fieldName: "cc_allocation",
                        value: e.target.value
                    })}
                />
            </div>
            <div>
                <label className="block mb-2 text-sm font-medium text-gray-900 dark:text-white">
                    compute canada project
                </label>
                <input
                    type="text"
                    className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                    placeholder="def-my-project"
                    value={dpFoldArgsView.args.cc_project || ""}
                    onChange={e => dpFoldArgsDispatcher({
                        name: "update",
                        fieldName: "cc_project",
                        value: e.target.value
                    })}
                />
            </div>

            <div>
                <DefaultButton caption={"Save"} isDisabled={! dpFoldArgsView.canSave}/>
            </div>
        </div>
}


export default DPFoldArgsEditor