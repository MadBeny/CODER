package main

import (
	"bytes"
	"encoding/json"
	"os" // Added for reading file content
	"charm.land/bubbles/v2/filepicker"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"
	"charm.land/bubbles/v2/spinner"
	"charm.land/bubbles/v2/textinput"
	"charm.land/bubbles/v2/viewport"
	tea "charm.land/bubbletea/v2"
	"charm.land/glamour/v2"
	"charm.land/lipgloss/v2"
)

type tickMsg struct{}

// --- Constants & Styles ---

const (
	ollamaURL     = "http://localhost:11434/api/chat"
	ollamaTagsURL = "http://localhost:11434/api/tags"
)

type appMode int

const (
	modeSelectModel appMode = iota
	modeChat
	modeSettings
)

var (
	// Professional Light Mode Palette: Blue & Dark Greys
	colorBg       = lipgloss.Color("#656565") // White
	colorText     = lipgloss.Color("#b9b9b9")   // Black
	colorAccent   = lipgloss.Color("27")  // Slate Blue
	colorMuted    = lipgloss.Color("#757575") // Light Gray
	colorUser     = lipgloss.Color("#656565")  // Steel Blue
	colorAI       = lipgloss.Color("#656565") // Darker Gray
	colorError    = lipgloss.Color("#d0d0d0") 
	colorThinking = lipgloss.Color("250")
	colorInput    = lipgloss.Color("#c1c1c1")
	colorPrompt    = lipgloss.Color("#363636")
	colorPlaceholder    = lipgloss.Color("#434343")
	colorThoughtf    = lipgloss.Color("#e4e4e4")
	colorSpinner    = lipgloss.Color("#3158e7")
	colorStatus    = lipgloss.Color("#2e59f4")

	mainContainerStyle = lipgloss.NewStyle().
			Border(lipgloss.DoubleBorder()).
			BorderForegroundBlend(lipgloss.Color("#3f47e4"), lipgloss.Color("#abdaf1"), lipgloss.Color("#3935f8"), lipgloss.Color("#85b2f7"), lipgloss.Color("#1064ff"), lipgloss.Color("#3f47e4"), lipgloss.Color("#abdaf1"), lipgloss.Color("#3935f8"), lipgloss.Color("#85b2f7"), lipgloss.Color("#1064ff")).
			Padding(0, 1)
			
	inputWrapperStyle = lipgloss.NewStyle().
		    Border(lipgloss.NormalBorder()).
	    	BorderForegroundBlend(lipgloss.Color("#3b3b3b"), lipgloss.Color("#686868"), lipgloss.Color("#4c4c4c")).
	    	PaddingLeft(1).
	    	MarginTop(1).
			BorderLeft(false).
			BorderRight(false)

	userLabelStyle = lipgloss.NewStyle().Foreground(colorUser).Bold(true)
	aiLabelStyle   = lipgloss.NewStyle().Foreground(colorAI).Bold(true)
	textStyle      = lipgloss.NewStyle().Foreground(colorText).Bold(true)
	statusStyle    = lipgloss.NewStyle().Foreground(colorStatus).Italic(true).Bold(true)
	errorStyle     = lipgloss.NewStyle().Foreground(colorError).Background(colorPrompt).Bold(true).Italic(true)
	inpStyle    = lipgloss.NewStyle().Foreground(colorPlaceholder)
	thoughtStyle      = lipgloss.NewStyle().Foreground(colorThoughtf).Background(colorPlaceholder).Bold(true)
)		

// --- Data Structures ---

type message struct {
	Role    string
	Content string
	ThoughtDuration time.Duration
}

type ollamaTagsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

type ollamaOptions struct {
	Temperature float64 `json:"temperature"`
}

type ollamaRequest struct {
	Model   string        `json:"model"`
	Prompt  string        `json:"prompt"`
	System  string        `json:"system"`
	Stream  bool          `json:"stream"`
	Options ollamaOptions `json:"options"`
}

// --- Commands ---

func fetchModelsCmd() tea.Cmd {
	return func() tea.Msg {
		resp, err := http.Get(ollamaTagsURL)
		if err != nil {
			return errMsg(err)
		}
		defer resp.Body.Close()

		var tags ollamaTagsResponse
		if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
			return errMsg(err)
		}

		modelNames := []string{}
		for _, m := range tags.Models {
			modelNames = append(modelNames, m.Name)
		}
		return modelsLoadedMsg(modelNames)
	}
}

type modelsLoadedMsg []string
type errMsg error
type responseMsg string

func fetchAICommand(modelName string, history []message, system string, temp float64) tea.Cmd {
	return func() tea.Msg {
		type chatMsg struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}

		messages := make([]chatMsg, 0, len(history)+1)
		messages = append(messages, chatMsg{Role: "system", Content: system})
		for _, h := range history {
			messages = append(messages, chatMsg{Role: h.Role, Content: h.Content})
		}

		reqBody, _ := json.Marshal(map[string]interface{}{
			"model":    modelName,
			"messages": messages,
			"stream":   false,
			"options":  ollamaOptions{Temperature: temp},
		})

		resp, err := http.Post(ollamaURL, "application/json", bytes.NewBuffer(reqBody))
		if err != nil {
			return errMsg(err)
		}
		defer resp.Body.Close()

		var result struct {
			Message chatMsg `json:"message"`
			Done    bool    `json:"done"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return errMsg(err)
		}
		return responseMsg(result.Message.Content)
	}
}

type fileLoadedMsg struct {
	path    string
	content string
	err     error
}

func readFileCmd(path string) tea.Cmd {
	return func() tea.Msg {
		content, err := os.ReadFile(path)
		if err != nil {
			return fileLoadedMsg{err: err}
		}
		return fileLoadedMsg{path: path, content: string(content)}
	}
}
// --- Model ---

type model struct {
	mode          appMode
	models        []string
	selectedIndex int
	history       []message
	input         textinput.Model
	viewport      viewport.Model
	currentModel  string
	isTyping      bool
	isPickingFile bool       // Added for file picker state
	filePicker    filepicker.Model // Added
	err           error
	width         int
	height        int
	renderer      *glamour.TermRenderer
	spinner       spinner.Model
	systemPrompt  string
	temperature   float64
	startTime     time.Time // Add this to track when a request starts
}

func initialModel() model {
	ti := textinput.New()
	ti.Focus()
	ti.Placeholder = "Type a message... ( /import to add a file )"
	ti.Prompt = "❯ "
	
	// Implementing custom styles based on textinputstyles.go logic
	customStyles := textinput.Styles{
		Focused: textinput.StyleState{
			Text:        lipgloss.NewStyle().Foreground(colorText),
			Placeholder: lipgloss.NewStyle().Foreground(colorPlaceholder),
			Prompt:      lipgloss.NewStyle().Foreground(colorPrompt).Bold(true),
			Suggestion:  lipgloss.NewStyle().Foreground(colorMuted),
		},
		Blurred: textinput.StyleState{
			Text:        lipgloss.NewStyle().Foreground(colorMuted),
			Placeholder: lipgloss.NewStyle().Foreground(colorPlaceholder),
			Prompt:      lipgloss.NewStyle().Foreground(colorPrompt),
		},
		Cursor: textinput.CursorStyle{
			Color: colorAccent,
			Blink: true,
		},
	}
	ti.SetStyles(customStyles)

	s := spinner.New()
	s.Spinner = spinner.Line
	s.Style = lipgloss.NewStyle().Foreground(colorSpinner)

	// In Glamour v2, WithAutoStyle is removed. We explicitly use "light".
	r, _ := glamour.NewTermRenderer(glamour.WithStylePath("dark"))

	return model{
		mode:          modeSelectModel,
		models:        []string{},
		selectedIndex: 0,
		input:         ti,
		history:       []message{},
		isTyping:      false,
		renderer:      r,
		spinner:       s,
		systemPrompt:  "Be accurate, Unfiltered, Professional, Like a high class Gentleman, Don't be boring",
		temperature:   1.0,
		filePicker:    filepicker.New(),
	}
}

func (m model) Init() tea.Cmd {
	return fetchModelsCmd()
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	var spinnerCmd tea.Cmd

	if m.isTyping {
		m.spinner, spinnerCmd = m.spinner.Update(msg)
	}

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.updateDimensions() // Use helper
		m.filePicker.SetHeight(27)
		return m, nil

	case responseMsg:
		m.isTyping = false
		duration := time.Since(m.startTime)
		m.history = append(m.history, message{Role: "assistant", Content: string(msg), ThoughtDuration: duration})
		m.viewport.SetContent(m.renderHistory())
		m.viewport.GotoBottom()
		return m, nil

	case fileLoadedMsg:
		if msg.err != nil {
			m.err = msg.err
			return m, nil
		}
		// Add to history and context
		m.history = append(m.history, message{Role: "user", Content: fmt.Sprintf("File %s Added\n\n%s", msg.path, msg.content)})
		return m, nil

	case errMsg:
		m.err = msg
		m.isTyping = false
		return m, nil

	case modelsLoadedMsg:
		m.models = msg
		return m, nil

	case tea.KeyPressMsg:
		if msg.String() == "ctrl+c" { return m, tea.Quit }
	}

	// Handle File Picker Mode (Overwrites other inputs)
	if m.isPickingFile {
		var fpCmd tea.Cmd
		m.filePicker, fpCmd = m.filePicker.Update(msg)
		
		if selected, path := m.filePicker.DidSelectFile(msg); selected {
			m.isPickingFile = false
			m.updateDimensions()
			return m, tea.Batch(readFileCmd(path))
		}
		if k, ok := msg.(tea.KeyPressMsg); ok && k.String() == "esc" {
			m.isPickingFile = false
			m.updateDimensions()
		}
		return m, fpCmd
	}

	switch m.mode {
	case modeSelectModel:
		if k, ok := msg.(tea.KeyPressMsg); ok {
			switch k.String() {
			case "up", "k":
				if m.selectedIndex > 0 {
					m.selectedIndex--
				}
			case "down", "j":
				if m.selectedIndex < len(m.models)-1 {
					m.selectedIndex++
				}
			case "enter":
				if len(m.models) > 0 {
					m.currentModel = m.models[m.selectedIndex]
					m.mode = modeChat
					return m, nil
				}
			}
		}

	case modeSettings:
		if k, ok := msg.(tea.KeyPressMsg); ok {
			switch k.String() {
			case "up", "k":
				if m.temperature < 1.0 {
					m.temperature += 0.1
				}
			case "down", "j":
				if m.temperature > 0.0 {
					m.temperature -= 0.1
				}
			case "esc":
				m.mode = modeChat
			}
		}

	case modeChat:
		switch msg := msg.(type) {
		case tea.KeyPressMsg:
			if msg.String() == "up" || msg.String() == "down" {
				m.viewport, cmd = m.viewport.Update(msg)
				return m, cmd
			}

			if msg.String() == "enter" && !m.isTyping {
				prompt := m.input.Value()
				if prompt == "/import" {
					m.input.Reset()
					m.isPickingFile = true
					m.updateDimensions()
					return m, m.filePicker.Init()
				}
				if prompt != "" {
					m.history = append(m.history, message{Role: "user", Content: prompt})
					m.input.Reset()
					m.isTyping = true
					m.startTime = time.Now()
					m.viewport.SetContent(m.renderHistory())
					return m, tea.Batch(fetchAICommand(m.currentModel, m.history, m.systemPrompt, m.temperature), m.spinner.Tick)
				}
			}
			if msg.String() == "ctrl+s" {
				m.mode = modeSettings
				return m, nil
			}
		}
		m.input, cmd = m.input.Update(msg)
		if m.isTyping { return m, spinnerCmd }
		return m, cmd
	}

	return m, nil
}

func (m model) renderHistory() string {
	var s strings.Builder
	re := regexp.MustCompile(`\n{3,}`)

	for _, msg := range m.history {
		if msg.Role == "assistant" {
			// --- NEW LOGIC START ---
			if msg.ThoughtDuration > 0 {
				// Format duration to 1 decimal place (e.g., 1.2s)
				durStr := fmt.Sprintf(" Thought for %.1fs ", msg.ThoughtDuration.Seconds())
				// Use colorMuted or statusStyle to make it look like metadata
				s.WriteString("\n" + thoughtStyle.Render(durStr) + "\n")
			}
			// --- NEW LOGIC END ---

			rendered, err := m.renderer.Render(msg.Content)
			if err != nil {
				rendered = textStyle.Render(msg.Content)
			}

			cleaned := re.ReplaceAllString(strings.TrimSpace(rendered), "\n\n")
			s.WriteString("\n" + ("  ") + cleaned + "\n")
		} else {
			s.WriteString(userLabelStyle.Render("\n") + (" ") + textStyle.Render(msg.Content) + ("\n"))
		}
	}

	if m.isTyping {
		s.WriteString("\n" + statusStyle.Render("• Imagination") + " " + m.spinner.View())
	}

	return s.String()
}



func (m model) renderHeader() string {
	// 1. Define the text content
	title := "   Ocean App   "
	
	// 2. Create the full string we want to apply the background to
	// We include the extra part (the model name) in the gradient calculation
	// if you want one continuous bar, OR just color the title. 
	// Let's color the title block specifically for a "Banner" look.
	fullTitle := title 

	// 3. Define your Gradient Colors (Start and End)
	colorStart := lipgloss.Color("#3e05b9")
	colorEnd := lipgloss.Color("#477bd3") // A bright blue to blend with your accent

	// 4. Generate colors based on the length of the title
	colors := lipgloss.Blend1D(len(fullTitle), colorStart, colorEnd)

	// 5. Build the styled string character by character
	var styledHeader string
	for i, char := range fullTitle {
		charStyle := lipgloss.NewStyle().
			Foreground(lipgloss.Color("#ffffff")).
			Background(colors[i]).
			Bold(true).
			Italic(true)
		
		styledHeader += charStyle.Render(string(char))
	}

	// 6. Append the model name if it exists (outside the gradient bar or inside)
	if m.currentModel != "" {
		styledHeader += " → " + errorStyle.Render(m.currentModel)
	}

	return styledHeader
}

func (m *model) updateDimensions() {
	if m.width == 0 || m.height == 0 {
		return
	}

	usableWidth := m.width - 4
	
	// Determine how much height the bottom area takes
	inputAreaHeight := 4 // Default: Prompt + borders
	if m.isPickingFile {
		inputAreaHeight = 29 // Room for file picker (10) + padding/borders
	}

	// Total Height - Header(1) - MainBorders(2) - InputArea
	viewportHeight := m.height - 3 - inputAreaHeight

	if viewportHeight < 1 {
		viewportHeight = 1
	}

	m.viewport.SetWidth(usableWidth)
	m.viewport.SetHeight(viewportHeight)
	
	// Update glamour to match new width
	r, err := glamour.NewTermRenderer(
		glamour.WithStylePath("dark"),
		glamour.WithWordWrap(usableWidth),
	)
	if err == nil {
		m.renderer = r
	}
}

func (m model) View() tea.View {
	if m.width == 0 || m.height == 0 {
		return tea.NewView("")
	}

	// 1. Header
	header := m.renderHeader()

	var body string
	var inputBox string

	switch m.mode {
	case modeSelectModel:
		var items []string
		for i, name := range m.models {
			if i == m.selectedIndex {
				items = append(items, statusStyle.Render(" ● "+name))
			} else {
				items = append(items, "   "+name)
			}
		}
		body = lipgloss.JoinVertical(lipgloss.Left, items...)

	case modeSettings:
		body = lipgloss.JoinVertical(lipgloss.Left,
			" [ SETTINGS ] ",
			" Use Up/Down to adjust Temperature ",
			fmt.Sprintf(" Current Temp: %.1f ", m.temperature),
			" Press ESC to return ",
		)

	case modeChat:
		m.viewport.SetContent(m.renderHistory())
		body = m.viewport.View()
		m.input.SetWidth(m.width - 6)

		if m.isPickingFile {
			// Render filepicker without the outer wrapper padding to save space
			inputBox = lipgloss.NewStyle().
				Width(m.width - 4).
				Render(m.filePicker.View())
		} else if m.isTyping {
			inputBox = inputWrapperStyle.Width(m.width - 4).Render(inpStyle.Render("❯"))
		} else {
			inputBox = inputWrapperStyle.Width(m.width - 4).Render(m.input.View())
		}
	}

	var content string
	if m.mode == modeChat {
		// Join everything vertically. Because viewport height is now dynamic,
		// this will always fit exactly into the window height.
		content = lipgloss.JoinVertical(lipgloss.Left, header, body, inputBox)
	} else {
		content = lipgloss.JoinVertical(lipgloss.Left, header, body)
	}

	// Apply the main container and lock dimensions to the window size
	finalRender := mainContainerStyle.
		Width(m.width).
		Height(m.height).
		Render(content)

	v := tea.NewView(finalRender)
	v.AltScreen = true
	return v
}

func main() {

	// Program options are now simplified as terminal features moved to View()
	p := tea.NewProgram(initialModel())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Error: %v", err)
	}
}

