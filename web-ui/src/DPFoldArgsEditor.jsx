
import React, {useReducer, useEffect} from 'react'
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
                    cc_cluster: action.args.cc_cluster,
                    cc_allocation: action.args.cc_allocation,
                    cc_project: action.args.cc_project
                },
                customFilesStatus: null,
                isDirty: false,
                canSave: false
            }
        }
        case "updateCustomFileStatus": {

            return {
                ...state,
                customFilesStatus: {
                    //name:
                    //exists
                    //isValid
                    ...action.status
                },
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

    useEffect(() => {

        //if(dpFoldArgsView.customFilesStatus === null) {return}

        api.getDPFoldCustomFilesStatus(pipelineInstance.pid).then(s => {
            dpFoldArgsDispatcher({
                name: "updateCustomFileStatus",
                status: s
            })
        })

    }, [])


    const saveArgs = () => {
        debugger
        api.savePipelineInstanceArgs(pipelineInstance.pid, dpFoldArgsView.args).then(() =>
            pipelineArgsDispatcher({name: "saved"})
        )
    }

    const errorPanel = () => {

        if(dpFoldArgsView.customFilesStatus && dpFoldArgsView.customFilesStatus.exists) {
            if (dpFoldArgsView.customFilesStatus.isValid) {
                return null
            }

            return <pre>
                {dpFoldArgsView.customFilesStatus.errors}
            </pre>

        }

        return null
    }

    const sampleSheetStatusDisplay = () => {

        if(dpFoldArgsView.customFilesStatus === null) {
            return "[...status loading]"
        }

        if(dpFoldArgsView.customFilesStatus.exists) {
            if(dpFoldArgsView.customFilesStatus.isValid) {
                return "[valid]"
            }
            return "[has errors]"
        }
        else {
            return "[file missing]"
        }

    }

    const sampleSheetField = () => {

        if(dpFoldArgsView.customFilesStatus === null) {
            return "...status loading"
        }

        if(dpFoldArgsView.customFilesStatus.exists) {
            return "samplesheet.tsv"
        }

        return ""

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
                <label className="block mb-2 text-sm font-medium text-gray-900 dark:text-white">
                    sample sheet { sampleSheetStatusDisplay() }
                </label>
                <input
                    type="text"
                    className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500"
                    placeholder="samplesheet.tsv"
                    value={sampleSheetField()}
                    disabled={true}
                />
                {errorPanel()}
            </div>

            <div>
                <DefaultButton
                    caption={"Save"}
                    isDisabled={! dpFoldArgsView.canSave}
                    onClick={saveArgs}
                />
            </div>
        </div>
}


export default DPFoldArgsEditor