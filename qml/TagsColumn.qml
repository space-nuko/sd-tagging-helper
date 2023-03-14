import QtQuick 2.15
import QtQuick.Controls 2.15
import QtGraphicalEffects 1.12

Item {
    id: root
    property var altLayoutMode: false

    signal tagAdded()
    signal tagDeleted()
    signal focusRelease()
    
    function addFavourites() {
        for(var i = 0; i < backend.favourites.length; i++) {
            addFavourite(i)
        }
    }

    function addFavourite(i) {
        var tag = backend.favourites[i]
        if(!backend.tags.includes(tag)) {
            backend.addTag(tag)
            root.tagAdded()
        } else {
            backend.deleteTagByName(tag)
            root.tagDeleted()
        }
    }

    function addDdb() {
        backend.addTags(backend.ddb)
    }

    Search {
        id: searchBox
        anchors.top: parent.top
        height: 30
        anchors.left: parent.left
        anchors.right: parent.right
        
        onFocusRelease: {
            root.focusRelease()
        }

        onTextChanged: {
            backend.search(text)
            searchTagsList.selectFirst()
        }

        onEnter: {
            searchTagsList.selectEnter()
            clear()
            searchTagsList.selectFirst()
        }

        onUp: {
            searchTagsList.selectUp()
        }

        onDown: {
            searchTagsList.selectDown()
        }
    }

    Rectangle {
        id: searchTags
        color: "#202020"
        anchors.top: searchBox.bottom
        anchors.bottom: sugDivider.top
        anchors.left: searchBox.left
        anchors.right: searchBox.right
        clip: true

        Column {
            anchors.centerIn: parent
            width: parent.width
            UniformText {
                id: addText
                visible: backend.results.length == 0
                text: "Add <font color=\"#fff\">" + searchBox.text + "</font> to tags?"
                anchors.horizontalCenter: parent.horizontalCenter
                
                width: parent.width - 20
                wrapMode: Text.WordWrap
                horizontalAlignment: Text.AlignHCenter
                font.pointSize: 11
                font.bold: false
                color: "#aaa"
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
            }
            IconButton {
                id: addButton
                
                anchors.horizontalCenter: parent.horizontalCenter
                visible: addText.visible
                width: 30
                height: 30
                icon: "qrc:/icons/plus.svg"
                tooltip: "Add tag"
                color: "transparent"
                onPressed: {
                    backend.addCustomTag(searchBox.text)
                    var search = searchBox.text
                    searchBox.clear()
                    searchBox.text = search
                }
            }
        }

        Tags {
            id: searchTagsList
            index: 2
            model: backend.results
            anchors.fill: parent
            moveEnabled: false

            function getOverlay(tag, index, model) {
                return backend.tags.includes(tag) ? "#77000000" : "#00000000"
            }

            onDoublePressed: {
                if(!backend.tags.includes(tag)) {
                    backend.addTag(tag)
                    root.tagAdded()
                } else {
                    backend.deleteTagByName(tag)
                    root.tagDeleted()
                }
            }

            onModelChanged: {
                populate()
            }
        }
    }

    Rectangle {
        z:10
        id: sugDivider
        anchors.left: parent.left
        anchors.right: parent.right
        height: 5
        property int minY: 30
        property int maxY: parent.height - 10
        color: "#404040"

        Component.onCompleted: {
            y = parent.height/3
        }

        MouseArea {
            anchors.fill: parent
            hoverEnabled: true
            onPositionChanged: {
                if(pressedButtons) {
                    sugDivider.y = Math.min(sugDivider.maxY, Math.max(sugDivider.minY, sugDivider.y + mouseY))
                }
            }
        }

        onMaxYChanged: {
            sugDivider.y = Math.min(sugDivider.maxY, Math.max(sugDivider.minY, sugDivider.y))
        }
    }

    Connections {
        target: backend
        function onListEvent(event) {
            // handle entering the search box and switching freq/ddb mode
            if(event == 2) {
                if(backend.activeList == 1) {
                    searchBox.gainFocus()
                }
                if(backend.activeList != 1 && searchBox.hasFocus) {
                    root.focusRelease()
                }
                if(!root.altLayoutMode) {
                    if(backend.activeList == 3 && !backend.showingFrequent) {
                        backend.showFrequent()
                    }
                    if(backend.activeList == 5 && backend.showingFrequent) {
                        if(backend.ddb.length > 0) {
                            backend.showDDB()
                        } else {
                            backend.doListEvent(3)
                        }
                    }
                }
            }
        }
    }

    Rectangle {
        id: sugLabel
        color: "#303030"
        anchors.top: sugDivider.bottom
        height: 30
        anchors.left: sugDivider.left
        anchors.right: sugDivider.right
        UniformText {
            text: "Suggestions"
            font.pointSize: 11
            leftPadding: 8
            rightPadding: 16
            font.bold: false
            color: "white"
            verticalAlignment: Text.AlignVCenter
            width: Math.min(parent.width, implicitWidth)
            elide: Text.ElideRight
            anchors.left: parent.left
            anchors.top: parent.top
            anchors.bottom: parent.bottom
        }

        DDBButton {
            id: ddbButton
            visible: !altLayoutMode
            anchors.right: parent.right
            anchors.top: parent.top
            height: parent.height
            width: height
        }

        IconButton {
            visible: !altLayoutMode && !backend.showingFrequent
            anchors.right: ddbButton.left
            anchors.top: parent.top
            height: parent.height
            width: height
            icon: "qrc:/icons/bookshelf.svg"
            tooltip: "Show frequently used"
            color: "transparent"
            onPressed: {
                backend.showFrequent()
            }
        }
    }

    Rectangle {
        id: sugTags
        color: "#202020"
        anchors.top: sugLabel.bottom
        anchors.bottom: favDivider.top
        anchors.left: sugDivider.left
        anchors.right: sugDivider.right
        clip: true

        Tags {
            id: sugTagsList
            index: !altLayoutMode && !backend.showingFrequent ? 5 : 3
            model: !altLayoutMode && !backend.showingFrequent ? backend.ddb : backend.frequent
            anchors.fill: parent
            moveEnabled: false

            onIndexChanged: {
                if(index == backend.activeList) {
                    sugTagsList.selectFirst()
                }
            }

            function getOverlay(tag, index, model) {
                return backend.tags.includes(tag) ? "#77000000" : "#00000000"
            }

            onDoublePressed: {
                if(!backend.tags.includes(tag)) {
                    backend.addTag(tag)
                    root.tagAdded()
                } else {
                    backend.deleteTagByName(tag)
                    root.tagDeleted()
                }
            }

            onModelChanged: {
                populate()
            }
        }
    }

    Rectangle {
        z: 10
        id: favDivider
        anchors.left: parent.left
        anchors.right: parent.right
        height: 5

        y: parent.height - offset

        property var offset: parent.height/3
        property int minY: 5
        property int maxY: parent.height-(sugDivider.y + 5)
        color: "#404040"

        Component.onCompleted: {
            offset = parent.height/3
        }

        MouseArea {
            anchors.fill: parent
            hoverEnabled: true
            onPositionChanged: {
                if(pressedButtons) {
                    favDivider.offset = Math.max(favDivider.minY, Math.min(favDivider.maxY, root.height - (favDivider.y + mouseY)))
                }
            }
        }

        onMaxYChanged: {
            favDivider.offset = Math.max(favDivider.minY, Math.min(favDivider.maxY, favDivider.offset))
        }
    }

    

    Rectangle {
        id: favLabel
        color: "#303030"
        anchors.top: favDivider.bottom
        height: 30
        anchors.left: favDivider.left
        anchors.right: favDivider.right
        UniformText {
            text: "Favourites"
            font.pointSize: 11
            leftPadding: 8
            rightPadding: 16
            font.bold: false
            color: "white"
            verticalAlignment: Text.AlignVCenter
            width: Math.min(parent.width, implicitWidth)
            elide: Text.ElideRight
            anchors.left: parent.left
            anchors.top: parent.top
            anchors.bottom: parent.bottom
        }

        IconButton {
            anchors.right: parent.right
            anchors.top: parent.top
            height: parent.height
            width: height
            icon: "qrc:/icons/plus.svg"
            tooltip: "Add all"
            color: "#303030"
            onPressed: {
                root.addFavourites()
            }
            onContextMenu: {
                addContextMenu.popup()
            }
        }

        ContextMenu {
            id: addContextMenu

            Action {
                id: prefixTags
                text: "Add tags to start?" 

                checkable: true
                
                onCheckedChanged: {
                    backend.setPrefixingTags(checked)
                    checked = backend.prefixingTags
                }

                Component.onCompleted: {
                    checked = backend.prefixingTags
                }
            }

            onClosed: {
                keyboardFocus.forceActiveFocus()
            }
        }
    }

    Rectangle {
        id: favTags
        color: "#202020"
        anchors.top: favLabel.bottom
        anchors.bottom: parent.bottom
        anchors.left: favDivider.left
        anchors.right: favDivider.right
        clip: true

        Tags {
            id: favTagsList
            index: 4
            model: backend.favourites
            anchors.fill: parent

            function getOverlay(tag, index, model) {
                return backend.tags.includes(tag) ? "#77000000" : "#00000000"
            }

            onDoublePressed: {
                if(!backend.tags.includes(tag)) {
                    backend.addTag(tag)
                    root.tagAdded()
                } else {
                    backend.deleteTagByName(tag)
                    root.tagDeleted()
                }
            }
       
            onMoved: {
                backend.moveFavourite(from, to)
            }

            onModelChanged: {
                populate()
            }
        }
    }
}
